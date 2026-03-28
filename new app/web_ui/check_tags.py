import sys

def check_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    div_open = content.count('<div')
    div_close = content.count('</div')
    print(f"DIV Open: {div_open}, Close: {div_close}")
    
    main_open = content.count('<main')
    main_close = content.count('</main')
    print(f"MAIN Open: {main_open}, Close: {main_close}")
    
    section_open = content.count('<section')
    section_close = content.count('</section')
    print(f"SECTION Open: {section_open}, Close: {section_close}")

if __name__ == "__main__":
    check_html(sys.argv[1])
