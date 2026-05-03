#!/usr/bin/env python3
"""
Convert ChatGPT exported conversations to Obsidian Copilot format.
"""

import json
import os
import re
import unicodedata
import html
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default model key for frontmatter
DEFAULT_MODEL = 'openai/gpt-oss-120b|openrouterai'

def extract_json_from_html(html_file: str) -> Tuple[Optional[List], Optional[Dict]]:
    """
    Extract jsonData and assetsJson variables from chat.html file.
    Returns (conversations_list, assets_dict) or (None, None) if extraction fails.
    """
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract jsonData variable - find the array and parse it carefully
        json_data = None
        json_match = re.search(r'var\s+jsonData\s*=\s*(\[)', content)
        if json_match:
            start_pos = json_match.start(1)
            # Find matching closing bracket
            bracket_count = 0
            in_string = False # boolean flag that tracks whether the parser is currently inside a JSON string literal.
            escape_next = False # boolean flag that tracks whether the next character is escaped with a backslash.
            for i in range(start_pos, len(content)):
                char = content[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
				# Handle escaped quotes (\") inside strings so they don't incorrectly toggle the in_string flag
                if char == '\\':
                    escape_next = True
                    continue
                
				# Prevent counting brackets/braces that appear inside string values of JSON
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_str = content[start_pos:i+1]
                            json_data = json.loads(json_str)
                            break
        
        # Extract assetsJson variable
        assets_data = None
        assets_match = re.search(r'var\s+assetsJson\s*=\s*(\{)', content)
        if assets_match:
            start_pos = assets_match.start(1)
            # Find matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False
            for i in range(start_pos, len(content)):
                char = content[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = content[start_pos:i+1]
                            assets_data = json.loads(json_str)
                            break
        
        if json_data is None:
            print(f"✗ Could not find jsonData in {html_file}")
            return None, None
        
        return json_data, assets_data if assets_data else {}
    
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error in {html_file}: {e}")
        return None, None
    except Exception as e:
        print(f"✗ Error reading {html_file}: {e}")
        return None, None

def epoch_to_timestamp(epoch: float) -> str:
    """Convert Unix epoch to YYYY/MM/DD HH:MM:SS format."""
    dt = datetime.fromtimestamp(epoch)
    return dt.strftime('%Y/%m/%d %H:%M:%S')

def epoch_to_filename_date(epoch: float) -> str:
    """Convert Unix epoch to YYYYMMDD_HHMMSS format for filename."""
    dt = datetime.fromtimestamp(epoch)
    return dt.strftime('%Y%m%d_%H%M%S')

def sanitize_filename(text: str, max_words: int = 10) -> str:
    """
    Extract first N words and return a cleaned title preserving spaces.
    Removes special characters but keeps spaces so titles remain human-readable.
    """
    # Remove only path/divider characters that are invalid in filenames of Obsidian
    # Keep other punctuation characters so titles remain human-readable.
    text = re.sub(r'[\\/:]', '', text)
    # Normalize whitespace and split into words, taking first max_words
    words = re.sub(r'\s+', ' ', text).strip().split()[:max_words]
    # Join with single spaces and trim
    filename = ' '.join(words)
    # Collapse multiple spaces (defensive) and strip
    filename = re.sub(r'\s+', ' ', filename).strip()
    return filename

def get_conversation_messages(conversation: Dict) -> List[Dict]:
    """
    Extract messages from conversation mapping by traversing from current_node.
    Returns list of {author, parts, create_time} dicts.
    """
    messages = []
    mapping = conversation.get('mapping', {})

    # Iterate all nodes in mapping so we include every branch
    for node_id, node in mapping.items():
        if not node:
            continue
        message = node.get('message')
        if not message or not message.get('content') or not message['content'].get('parts'):
            continue

        author_role = message.get('author', {}).get('role')

        # Skip system messages unless they're marked as user system messages
        if author_role == 'system' and not message.get('metadata', {}).get('is_user_system_message'):
            continue

        # Map author roles
        if author_role in ('assistant', 'tool'):
            author = 'ai'
        elif author_role == 'system':
            author = 'user'
        else:
            author = 'user'

        content_type = message['content'].get('content_type')
        if content_type in ['text', 'multimodal_text']:
            parts = []
            for part in message['content']['parts']:
                if isinstance(part, str) and part.strip():
                    parts.append({'text': part})
                elif isinstance(part, dict):
                    ptype = part.get('content_type')
                    if ptype == 'audio_transcription':
                        parts.append({'transcript': part.get('text', '')})
                    elif ptype in ['audio_asset_pointer', 'image_asset_pointer', 'video_container_asset_pointer']:
                        parts.append({'asset': part})
                    elif ptype == 'real_time_user_audio_video_asset_pointer':
                        if part.get('audio_asset_pointer'):
                            parts.append({'asset': part['audio_asset_pointer']})
                        if part.get('video_container_asset_pointer'):
                            parts.append({'asset': part['video_container_asset_pointer']})
                        for frame in part.get('frames_asset_pointers', []):
                            parts.append({'asset': frame})

            if parts:
                # Capture any content_references and search_result_groups from the message metadata
                # so we can replace matched tokens and append sources later.
                metadata = message.get('metadata', {})
                content_refs = metadata.get('content_references', [])
                search_result_groups = metadata.get('search_result_groups', [])
                messages.append({
                    'node_id': node_id, # For debugging purposes
                    'author': author,
                    'parts': parts,
                    'create_time': message.get('create_time'),
                    'content_references': content_refs,
                    'search_result_groups': search_result_groups
                })

    # Sort by create_time (oldest first). Put messages without timestamps at the end.
    def _sort_key(m):
        t = m.get('create_time')
        return (t is None, t if t is not None else 0)

    messages.sort(key=_sort_key)
    return messages

def format_message_parts(parts: List[Dict], assets_map: Dict[str, str],
                         content_references: List[Dict] | None = None,
                         search_result_groups: List[Dict] | None = None) -> str:
    """
    Format message parts into markdown text.
    Handles text, transcripts, and asset pointers (images/audio/video).
    """
    output = []
    
    # Helper to build display text from a content_reference
    def build_reference_markdown(ref: Dict) -> str:
        rtype = ref.get('type')
        # grouped_webpages (and model-predicted fallback): build markdown links from items list
        if rtype in ('grouped_webpages', 'grouped_webpages_model_predicted_fallback') and ref.get('items'):
            links = []
            for it in ref.get('items', []):
                title = it.get('title') or it.get('attribution') or it.get('url')
                url = it.get('url')
                snippet = it.get('snippet')
                if url:
                    if title:
                        link_text = f"[{title}]({url})"
                    else:
                        link_text = f"[{url}]({url})"
                    if snippet:
                        link_text += f" *{snippet}*"
                    links.append(link_text)
                
                # Add supporting_websites if present within this item (format same as items)
                supporting = it.get('supporting_websites', [])
                if supporting:
                    for sw in supporting:
                        sw_title = sw.get('title') or sw.get('attribution') or sw.get('url')
                        sw_url = sw.get('url')
                        sw_snippet = sw.get('snippet')
                        if sw_url:
                            if sw_title:
                                sw_link = f"[{sw_title}]({sw_url})"
                            else:
                                sw_link = f"[{sw_url}]({sw_url})"
                            if sw_snippet:
                                sw_link += f" *{sw_snippet}*"
                            links.append(sw_link)
            
            if links:
                return f"({' '.join(links)})"
        # image_group: format images and links as a markdown table (images row, links row)
        if rtype == 'image_group' and ref.get('images'):
            img_cells = []
            title_cells = []
            for im in ref.get('images', []):
                img_result = im.get('image_result', {})
                content_url = img_result.get('content_url')
                if content_url:
                    # Build image embed with search query and width if available
                    search_query = im.get('image_search_query', '')
                    width = img_result.get('thumbnail_size', {}).get('width')
                    if width:
                        cell = f"![{search_query}|{width}]({content_url})"
                    else:
                        cell = f"![{search_query}]({content_url})"
                    # Escape any '|' characters so table columns are not broken
                    cell = cell.replace('|', '\\|')
                    img_cells.append(cell)
                else:
                    img_cells.append("")

                # Build title link if available
                title = img_result.get('title')
                url = img_result.get('url')
                if title and url:
                    tcell = f"[{title}]({url})"
                    tcell = tcell.replace('|', '\\|')
                    title_cells.append(tcell)
                else:
                    title_cells.append("")

            # Build markdown table: imgs row, separator, titles row
            if img_cells or title_cells:
                n = max(len(img_cells), len(title_cells))
                while len(img_cells) < n:
                    img_cells.append("")
                while len(title_cells) < n:
                    title_cells.append("")

                row1 = "\n| " + " | ".join(img_cells) + " |"
                sep = "| " + " | ".join(["---"] * n) + " |"
                row2 = "| " + " | ".join(title_cells) + " |"
                return "\n".join([row1, sep, row2])
        # image_v2: same as image_group but different field names (url, content_url, title, thumbnail_size.width)
        if rtype == 'image_v2' and ref.get('images'):
            img_cells = []
            title_cells = []
            for im in ref.get('images', []):
                # image_v2 elements: url, content_url, title, thumbnail_size.width
                content_url = im.get('content_url') or im.get('url')
                title = im.get('title')
                width = None
                thumb = im.get('thumbnail_size') or {}
                if isinstance(thumb, dict):
                    width = thumb.get('width')

                if content_url:
                    search_query = title or ''
                    if width:
                        cell = f"![{search_query}|{width}]({content_url})"
                    else:
                        cell = f"![{search_query}]({content_url})"
                    cell = cell.replace('|', '\\|')
                    img_cells.append(cell)
                else:
                    img_cells.append("")

                # title link
                url = im.get('url')
                if title and url:
                    tcell = f"[{title}]({url})"
                    tcell = tcell.replace('|', '\\|')
                    title_cells.append(tcell)
                else:
                    title_cells.append("")

            if img_cells or title_cells:
                n = max(len(img_cells), len(title_cells))
                while len(img_cells) < n:
                    img_cells.append("")
                while len(title_cells) < n:
                    title_cells.append("")

                row1 = "\n| " + " | ".join(img_cells) + " |"
                sep = "| " + " | ".join(["---"] * n) + " |"
                row2 = "| " + " | ".join(title_cells) + " |"
                return "\n".join([row1, sep, row2])
        # products: format an array of product-like entries as a markdown table
        if rtype == 'products' and ref.get('products'):
            prods = ref.get('products', [])
            img_cells = []
            title_cells = []
            for p in prods:
                # Prefer explicit title, then name
                title = p.get('title') or p.get('name') or ''
                url = p.get('url')

                # Try first product image if present
                image_url = None
                imgs = p.get('image_urls') or p.get('images')
                if isinstance(imgs, list) and imgs:
                    image_url = imgs[0]

                # build image cell
                if image_url:
                    cell = f"![{title}]({image_url})"
                    cell = cell.replace('|', '\\|')
                    img_cells.append(cell)
                else:
                    img_cells.append("")

                # build title/link cell (with merchant/price metadata)
                merchants = p.get('merchant') or p.get('merchants')
                price = p.get('price')
                if url:
                    main = f"[{title or url}]({url})"
                else:
                    main = title or ''
                meta = []
                if merchants:
                    meta.append(str(merchants))
                if price is not None:
                    meta.append(str(price))
                if meta and main:
                    main = f"{main} — {' | '.join(meta)}"

                # escape '|' in title cell
                title_cells.append(main.replace('|', '\\|'))

            # Build markdown table: imgs row, separator, titles row
            if img_cells or title_cells:
                # ensure equal length
                n = max(len(img_cells), len(title_cells))
                while len(img_cells) < n:
                    img_cells.append("")
                while len(title_cells) < n:
                    title_cells.append("")

                row1 = "\n| " + " | ".join(img_cells) + " |"
                sep = "| " + " | ".join(["---"] * n) + " |"
                row2 = "| " + " | ".join(title_cells) + " |"
                return "\n".join([row1, sep, row2])
        # product_entity: render image and title on separate lines
        if rtype == 'product_entity' and ref.get('product'):
            p = ref.get('product')
            # Prefer explicit title, then name
            title = p.get('title') or p.get('name') or ''
            url = p.get('url')

            # Try first product image if present
            image_url = None
            imgs = p.get('image_urls') or p.get('images')
            if isinstance(imgs, list) and imgs:
                image_url = imgs[0]

            merchants = p.get('merchant') or p.get('merchants')
            price = p.get('price')

            parts = []
            if image_url:
                parts.append(f"![{title}]({image_url})")

            # Build main product line with optional merchants/price
            if url:
                main = f"[{title or url}]({url})"
            else:
                main = title or ''

            meta = []
            if merchants:
                meta.append(str(merchants))
            if price is not None:
                meta.append(str(price))
            if meta and main:
                main = f"{main} — {' \\| '.join(meta)}"

            if main:
                parts.append(main)

            # Place image and title/link on the same line separated by a space
            return ' '.join([p for p in parts if p]) if parts else ''
        return ''

    # (sources footnote formatting is handled separately)

    # Prepare text parts for position-based replacement if references provide indices.
    # Collect indices of text parts and their content.
    text_part_indexes = [i for i, p in enumerate(parts) if 'text' in p]
    if text_part_indexes and content_references:
        # Gather refs that include start/end indices
        indexed_refs = [r for r in content_references if r.get('start_idx') is not None and r.get('end_idx') is not None]
        if indexed_refs:
            # Build a single full_text by concatenating all text parts in order
            texts = [parts[i]['text'] for i in text_part_indexes]
            full_text = ''.join(texts)

            # Some exports HTML-escape characters like '"' and '&' into
            # "&quot;" and "&amp;". The content_references' start_idx/end_idx
            # are measured against the unescaped text, so unescape here so
            # index-based replacements match the spans correctly.
            full_text = html.unescape(full_text)

            # Normalize: JSON already contains real newlines and unicode codepoints,
            # indices in refs are expected to match this representation.

            # Sort refs descending by start_idx to avoid shifting indices when replacing
            indexed_refs.sort(key=lambda r: r.get('start_idx', 0), reverse=True)
            for ref in indexed_refs:

                # If this is a hidden reference, remove the span without replacement
                if ref.get('type') == 'hidden':
                    start = int(ref.get('start_idx'))
                    end = int(ref.get('end_idx'))
                    if 0 <= start < end <= len(full_text):
                        full_text = full_text[:start] + full_text[end:]
                    continue

                # If grouped_webpages (or model-predicted fallback) has empty items, remove the span without replacement
                if ref.get('type') in ('grouped_webpages', 'grouped_webpages_model_predicted_fallback') and not ref.get('items'):
                    start = int(ref.get('start_idx'))
                    end = int(ref.get('end_idx'))
                    if 0 <= start < end <= len(full_text):
                        full_text = full_text[:start] + full_text[end:]
                    continue

                # Build replacement text based on type
                alt = build_reference_markdown(ref)

                start = int(ref.get('start_idx'))
                end = int(ref.get('end_idx'))
                # Validate indices
                if 0 <= start < end <= len(full_text) and alt:
                        # Optionally verify matched_text matches span. Normalize
                        # Unicode differences (non-breaking spaces, narrow no-break spaces,
                        # and hyphen-like characters) before comparing so small
                        # formatting differences don't prevent replacements.
                        def _normalize_for_compare(s: Optional[str]) -> Optional[str]:
                            if s is None:
                                return None
                            # NFC normalize, then map common non-breaking spaces to regular
                            s = unicodedata.normalize('NFC', s)
                            s = s.replace('\u00A0', ' ').replace('\u202F', ' ')
                            # Map a few hyphen-like characters to ASCII hyphen
                            for ch in ('\u2010', '\u2011', '\u2012', '\u2013', '\u2014'):
                                s = s.replace(ch, '-')
                            # Collapse any whitespace runs to a single space for robust matching
                            s = re.sub(r'\s+', ' ', s)
                            return s

                        matched = ref.get('matched_text')
                        # Ensure matched_text is unescaped as well so comparisons
                        # succeed when refs include characters that were HTML-escaped
                        # in the original parts.
                        if isinstance(matched, str):
                            matched = html.unescape(matched)
                        span = full_text[start:end]
                        if matched is None or _normalize_for_compare(matched) == _normalize_for_compare(span):
                            full_text = full_text[:start] + alt + full_text[end:]

            # After processing indexed refs, collect and append sources from search_result_groups and grouped_webpages
            sources = []
            seen_sources = set()
            def add_source(attribution: Optional[str], url: Optional[str], title: Optional[str], snippet: Optional[str]):
                if snippet:
                    snippet = snippet.replace('\n', ' ').replace('\r', ' ')
                key = None
                if url:
                    key = ('url', url.strip(), attribution or '', title or '', snippet or '')
                else:
                    key = ('text', (attribution or '').strip(), (title or '').strip(), (snippet or '').strip())
                if key in seen_sources:
                    return
                seen_sources.add(key)
                sources.append({
                    'attribution': attribution,
                    'url': url,
                    'title': title,
                    'snippet': snippet
                })

            if search_result_groups:
                for group in search_result_groups:
                    for entry in group.get('entries', []):
                        add_source(
                            entry.get('attribution'),
                            entry.get('url'),
                            entry.get('title'),
                            entry.get('snippet')
                        )
            for ref in content_references or []:
                if ref.get('type') == 'grouped_webpages':
                    for item in ref.get('items', []):
                        add_source(
                            item.get('attribution'),
                            item.get('url'),
                            item.get('title'),
                            item.get('snippet')
                        )
                        for sw in item.get('supporting_websites', []):
                            add_source(
                                sw.get('attribution') or sw.get('title'),
                                sw.get('url'),
                                sw.get('title'),
                                sw.get('snippet')
                            )
            if sources:
                lines = ['\n* Sources:']
                for s in sources:
                    attribution = s.get('attribution') or s.get('title') or s.get('url')
                    url = s.get('url')
                    title = s.get('title')
                    snippet = s.get('snippet')
                    if snippet:
                        snippet = snippet.replace('\n', ' ').replace('\r', ' ')
                    if url and attribution:
                        lines.append(f"\t* [{attribution}]({url})")
                        if title:
                            lines.append(f"\t\t**{title}**")
                        if snippet:
                            lines.append(f"\t\t{snippet}")
                full_text += '\n'.join(lines)
            
            # Replace original text parts with the modified full_text — put into first text part
            for idx, part_idx in enumerate(text_part_indexes):
                if idx == 0:
                    parts[part_idx]['text'] = full_text
                else:
                    # clear other parts since we've collapsed them
                    parts[part_idx]['text'] = ''

    for part in parts:
        if 'text' in part:
            # Replace escaped LaTeX display delimiters (\\[ ... \\]) with Obsidian display math ($$ ... $$)
            text = part['text']
            if isinstance(text, str) and text:
                text = text.replace('\\[', '$$').replace('\\]', '$$')
                text = html.unescape(text)
            output.append(text)
        elif 'transcript' in part:
            # Also replace escaped brackets in transcripts
            transcript = part['transcript']
            if isinstance(transcript, str) and transcript:
                if content_references:
                    for ref in content_references:
                        matched = ref.get('matched_text')
                        # For non-indexed refs, build alt from type/items if needed
                        alt = build_reference_markdown(ref)
                        if matched and alt:
                            transcript = transcript.replace(matched, alt)

                transcript = transcript.replace('\\[', '$$').replace('\\]', '$$')
                transcript = html.unescape(transcript)
            output.append(f"[Transcript]: {transcript}")
        elif 'asset' in part:
            asset_pointer = part['asset'].get('asset_pointer', '')
            if asset_pointer in assets_map:
                filename = assets_map[asset_pointer]
                # Use Obsidian image syntax for images
                if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    output.append(f"![[{filename}]]")
                else:
                    # For other files, just link them
                    output.append(f"[File]: [[{filename}]]")
            else:
                output.append("[File]: -Deleted-")
    
    return '\n'.join(output)

def convert_conversation_to_markdown(conversation: Dict, assets_map: Dict[str, str], model: str = DEFAULT_MODEL) -> tuple[str, str]:
    """
    Convert a single conversation to Obsidian Copilot markdown format.
    Returns (filename, markdown_content).
    """
    title = conversation.get('title', 'Untitled')
    messages = get_conversation_messages(conversation)
    
    if not messages:
        return None, None
    
    # Get create_time from first message so epoch frontmatter matches first message timestamp
    create_time = messages[0].get('create_time', 0)
    
    # Get first user message for filename
    first_user_msg = None
    for msg in messages:
        if msg['author'] == 'user':
            first_user_msg = msg
            break
    
    if not first_user_msg:
        return None, None
    
    # Extract first text part for filename
    first_text = ''
    for part in first_user_msg['parts']:
        if 'text' in part:
            first_text = part['text']
            break
    
    # Generate filename from conversation title
    topic_part = sanitize_filename(title, max_words=10)
    timestamp = epoch_to_filename_date(create_time)
    filename = f"{topic_part} @{timestamp}.md"
    
    # Build markdown content
    lines = []
    
    # YAML frontmatter
    lines.append('---')
    lines.append(f'epoch: {int(create_time) * 1000}')  # Convert to milliseconds (rounded to nearest second)
    lines.append(f'modelKey: "{model}"')
    lines.append(f'topic: "{title}"')
    lines.append('')
    lines.append('')
    lines.append('tags:')
    lines.append('  - copilot-conversation')
    lines.append('---')
    lines.append('')
    
    # Messages
    for msg in messages:
        author = msg['author']
        content = format_message_parts(msg['parts'], assets_map, msg.get('content_references', []), msg.get('search_result_groups', []))
        timestamp = epoch_to_timestamp(msg['create_time']) if msg.get('create_time') else ''
        
        if content.strip():  # Only add non-empty messages
            lines.append(f"**{author}**: {content}")
            if timestamp:
                lines.append(f"[Timestamp: {timestamp}]")
            lines.append('')
    
    return filename, '\n'.join(lines)

def unique_filepath(output_dir: Path, filename: str) -> Path:
    """Return a Path inside output_dir that does not yet exist by appending numbered suffixes.

    Example: topic.md -> topic.md, topic_1.md, topic_2.md
    """
    base = Path(filename).stem
    ext = Path(filename).suffix or '.md'
    candidate = output_dir / filename
    i = 1
    while candidate.exists():
        candidate = output_dir / f"{base}_{i}{ext}"
        i += 1
    return candidate

def main():
    """Main conversion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert ChatGPT exports to Obsidian Copilot format')
    parser.add_argument('json_file', nargs='?', help='Path to conversations.json file')
    parser.add_argument('--assets', help='Path to assets.json file (optional)')
    parser.add_argument('--html', help='Path to chat.html file (extracts jsonData and assetsJson variables)')
    parser.add_argument('--output-dir', default='./converted', help='Output directory for markdown files')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Model key for frontmatter')
    
    args = parser.parse_args()
    
    # Load conversations and assets
    conversations = None
    assets_map = {}
    
    if args.html:
        # Extract from HTML file
        conversations, assets_map = extract_json_from_html(args.html)
        if conversations is None:
            return
    elif args.json_file:
        # Load from JSON files
        with open(args.json_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Load assets mapping if provided
        if args.assets and os.path.exists(args.assets):
            with open(args.assets, 'r', encoding='utf-8') as f:
                assets_map = json.load(f)
    else:
        parser.print_help()
        print("\n✗ Error: Either provide json_file or use --html option")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each conversation
    converted_count = 0
    for conversation in conversations:
        filename, markdown = convert_conversation_to_markdown(conversation, assets_map, args.model)
        
        if filename and markdown:
            # ensure unique filename by auto-appending numeric suffixes when necessary
            output_path = unique_filepath(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"✓ Converted: {output_path.name}")
            converted_count += 1
        else:
            conv_id = conversation.get('id', 'unknown')
            print(f"✗ Skipped conversation {conv_id}: No valid messages")
    
    print(f"\n✓ Converted {converted_count} conversations to {output_dir}")
    print(f"\nNote: Copy images from your ChatGPT export to your Obsidian vault.")


if __name__ == '__main__':
    main()