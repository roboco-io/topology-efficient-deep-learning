#!/usr/bin/env python3
"""Q&A 문서 병합 스크립트."""

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def parse_frontmatter(content: str) -> Tuple[Dict, str]:
    """Frontmatter와 본문 분리."""
    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return {}, content

    frontmatter_text = match.group(1)
    body = match.group(2)

    # 간단한 YAML 파싱
    metadata = {}
    for line in frontmatter_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # 리스트 처리
            if value.startswith('[') and value.endswith(']'):
                value = [v.strip() for v in value[1:-1].split(',')]

            metadata[key] = value

    return metadata, body


def extract_question(body: str) -> str:
    """본문에서 질문 추출."""
    pattern = r'^#\s*Q:\s*(.+)$'
    match = re.search(pattern, body, re.MULTILINE)
    return match.group(1).strip() if match else "Unknown"


def load_qa_files(qa_dir: Path) -> List[Dict]:
    """Q&A 파일들 로드."""
    qa_files = []

    for file_path in sorted(qa_dir.glob('*.md')):
        if file_path.name.startswith('.'):
            continue

        content = file_path.read_text(encoding='utf-8')
        metadata, body = parse_frontmatter(content)

        qa_files.append({
            'file': file_path.name,
            'path': file_path,
            'metadata': metadata,
            'body': body,
            'question': extract_question(body),
            'category': metadata.get('category', 'general'),
            'date': metadata.get('date', ''),
            'tags': metadata.get('tags', []),
        })

    return qa_files


def generate_slug(question: str) -> str:
    """질문에서 slug 생성."""
    slug = question.lower()
    slug = re.sub(r'[^a-z0-9가-힣\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    return slug[:50]


def merge_by_category(qa_files: List[Dict]) -> str:
    """카테고리별로 병합."""
    category_order = ['topology', 'algebra', 'geometry', 'ml', 'statistics', 'general']
    category_names = {
        'topology': 'Topology (위상수학)',
        'algebra': 'Algebra (대수학)',
        'geometry': 'Geometry (기하학)',
        'ml': 'Machine Learning (머신러닝)',
        'statistics': 'Statistics (통계학)',
        'general': 'General (기타)',
    }

    # 카테고리별 그룹화
    by_category = {}
    for qa in qa_files:
        cat = qa['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(qa)

    # 문서 생성
    lines = [
        '# 개념 정리 (Concepts Reference)',
        '',
        '> 이 문서는 프로젝트 진행 중 학습한 수학적/기술적 개념들을 정리한 것입니다.',
        f'> 자동 생성됨: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '## 목차',
        '',
    ]

    # 목차 생성
    for cat in category_order:
        if cat in by_category:
            lines.append(f'- [{category_names.get(cat, cat)}](#{cat})')
            for qa in by_category[cat]:
                slug = generate_slug(qa['question'])
                lines.append(f'  - [{qa["question"]}](#{slug})')

    lines.append('')
    lines.append('---')
    lines.append('')

    # 본문 생성
    for cat in category_order:
        if cat not in by_category:
            continue

        lines.append(f'## {category_names.get(cat, cat)}')
        lines.append('')

        for qa in by_category[cat]:
            # 제목에서 # Q: 제거하고 본문 추가
            body = qa['body'].strip()

            # 메타 정보 추가
            if qa['tags']:
                tags = qa['tags'] if isinstance(qa['tags'], list) else [qa['tags']]
                tag_str = ', '.join(f'`{t}`' for t in tags)
                lines.append(f'*Tags: {tag_str}* | *Date: {qa["date"]}*')
                lines.append('')

            lines.append(body)
            lines.append('')
            lines.append('---')
            lines.append('')

    return '\n'.join(lines)


def merge_by_date(qa_files: List[Dict]) -> str:
    """날짜순으로 병합."""
    sorted_files = sorted(qa_files, key=lambda x: x['date'], reverse=True)

    lines = [
        '# 개념 정리 (Concepts Reference)',
        '',
        '> 이 문서는 프로젝트 진행 중 학습한 수학적/기술적 개념들을 정리한 것입니다.',
        f'> 자동 생성됨: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '---',
        '',
    ]

    for qa in sorted_files:
        body = qa['body'].strip()

        lines.append(f'*Category: `{qa["category"]}`* | *Date: {qa["date"]}*')
        lines.append('')
        lines.append(body)
        lines.append('')
        lines.append('---')
        lines.append('')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Q&A 문서 병합')
    parser.add_argument('--qa-dir', type=str, default='docs/qa',
                        help='Q&A 파일 디렉토리')
    parser.add_argument('--output', type=str, default='docs/CONCEPTS.md',
                        help='출력 파일 경로')
    parser.add_argument('--by-date', action='store_true',
                        help='날짜순 정렬 (기본: 카테고리별)')

    args = parser.parse_args()

    qa_dir = Path(args.qa_dir)
    output_path = Path(args.output)

    if not qa_dir.exists():
        print(f"Error: Q&A 디렉토리가 없습니다: {qa_dir}")
        return

    qa_files = load_qa_files(qa_dir)

    if not qa_files:
        print("Warning: Q&A 파일이 없습니다.")
        return

    print(f"Found {len(qa_files)} Q&A files")

    if args.by_date:
        content = merge_by_date(qa_files)
    else:
        content = merge_by_category(qa_files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding='utf-8')

    print(f"Merged document saved to: {output_path}")


if __name__ == '__main__':
    main()
