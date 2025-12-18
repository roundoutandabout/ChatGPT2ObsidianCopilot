#!/usr/bin/env python3
"""
Convert ChatGPT exported conversations to Obsidian Copilot format.
"""

import json
import os
import re
import html
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    # Remove special characters, keep only alphanumeric, spaces and hyphens
    text = re.sub(r'[^\w\s-]', '', text)
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
                messages.append({
					'node_id': node_id, # For debugging purposes
                    'author': author,
                    'parts': parts,
                    'create_time': message.get('create_time')
                })

    # Sort by create_time (oldest first). Put messages without timestamps at the end.
    def _sort_key(m):
        t = m.get('create_time')
        return (t is None, t if t is not None else 0)

    messages.sort(key=_sort_key)
    return messages

def format_message_parts(parts: List[Dict], assets_map: Dict[str, str]) -> str:
    """
    Format message parts into markdown text.
    Handles text, transcripts, and asset pointers (images/audio/video).
    """
    output = []
    
    for part in parts:
        if 'text' in part:
            # Replace escaped LaTeX display delimiters (\[ ... \]) with Obsidian display math ($$ ... $$)
            text = part['text']
            if isinstance(text, str) and text:
                text = text.replace('\\[', '$$').replace('\\]', '$$')
                text = html.unescape(text)
            output.append(text)
        elif 'transcript' in part:
            # Also replace escaped brackets in transcripts
            transcript = part['transcript']
            if isinstance(transcript, str) and transcript:
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

def convert_conversation_to_markdown(conversation: Dict, assets_map: Dict[str, str]) -> tuple[str, str]:
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
    filename = f"{topic_part}.md"
    
    # Build markdown content
    lines = []
    
    # YAML frontmatter
    lines.append('---')
    lines.append(f'epoch: {int(create_time) * 1000}')  # Convert to milliseconds (rounded to nearest second)
    lines.append('modelKey: "openai/gpt-oss-120b|openrouterai"')
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
        content = format_message_parts(msg['parts'], assets_map)
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
        filename, markdown = convert_conversation_to_markdown(conversation, assets_map)
        
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