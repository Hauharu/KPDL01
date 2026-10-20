import pandas as pd

def main():

    df = pd.read_csv('RestaurantDataset.csv', header=None, names=['Location', 'Type', 'Rating'], dtype=str)
    
    df['Location'] = df['Location'].fillna('').str.strip()
    df['Type'] = df['Type'].fillna('').str.strip()
    df['Rating'] = df['Rating'].fillna('').str.strip()
    
    locations = [loc for loc in df['Location'].unique() if loc]
    print("\n1. Các khu vực của New York xuất hiện trong danh sách:")
    for loc in sorted(locations):
        print(f"   - {loc}")
        
    print("\n" + "-"*50)
    
    types = [t for t in df['Type'].unique() if t]
    print("\n2. Các loại hình nhà hàng xuất hiện trong danh sách:")
    for t in sorted(types):
        print(f"   - {t}")
        
    print("\n" + "-"*50)
    
    ratings = [r for r in df['Rating'].unique() if r]
    print("\n3. Các hạng đánh giá xuất hiện trong danh sách:")
    for r in sorted(ratings):
        print(f"   - {r}")

if __name__ == "__main__":
    main()
