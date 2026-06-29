#!/usr/bin/env python3
import sys
import csv

def main():
    args = sys.argv

    #  ??trop tard?
    if len(args) < 2:
        print("Usage: ./describe.py <dataset.csv>")
        exit(1)

    file_train = args[1]
    try:
        with open(file_train, "r", encoding="utf-8") as file:
            reader = csv.reader(file)

            print("--- les premiers 3 lignes, change , a | ---")
            
            # 1. obtenir la premiere ligne comme header
            headers = next(reader)
            for row_idx in range(3):
                row_data = next(reader)
                
                print(f"\n[ 行号: {row_idx + 1} ]")
                print("-" * 30) # 打印一条分隔线
                
                # imprimer header et les donnes corresponds
                for col_idx, header in enumerate(headers):
                    value = row_data[col_idx] if col_idx < len(row_data) else "N/A"
                    
                    # 关键点：
                    # 1. header:<30 表示表头占用30个字符宽度，不足的用空格补齐
                    # 2. replace(" ", "-") 把表头后面的空格全部替换成横线 -
                    # 3. 紧接着打印 " | "
                    # 4. 最后打印数值 value
                    header_part = f"{header:<35}".replace(" ", "-")
                    
                    print(f"[{col_idx + 1:<2}] {header_part} | {value}")
                print("-" * 30)
            
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        exit(1) 


if __name__ == "__main__":
    main()
