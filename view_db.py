# view_db.py
import sqlite3
import os
from datetime import datetime

def view_database():
    # Check if database exists
    if not os.path.exists('schemes.db'):
        print("❌ schemes.db not found in current directory!")
        return
    
    print("🔍 Opening schemes.db...")
    conn = sqlite3.connect('schemes.db')
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    print(f"📊 Found {len(tables)} tables: {', '.join(tables)}")
    print("=" * 60)
    
    for table in tables:
        print(f"\n📋 Table: {table}")
        print("-" * 40)
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"Columns: {', '.join(columns)}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"Total rows: {count}")
        
        # Show first 5 rows
        if count > 0:
            cursor.execute(f"SELECT * FROM {table} LIMIT 5")
            rows = cursor.fetchall()
            print("First 5 rows:")
            for i, row in enumerate(rows, 1):
                print(f"  {i}. {row}")
        
        print("-" * 40)
    
    # Show some statistics
    print("\n📈 DATABASE STATISTICS")
    print("=" * 60)
    
    if 'interactions' in tables:
        # Total interactions
        cursor.execute("SELECT COUNT(*) FROM interactions")
        total = cursor.fetchone()[0]
        print(f"Total interactions: {total}")
        
        # Unique users
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM interactions")
        unique_users = cursor.fetchone()[0]
        print(f"Unique users: {unique_users}")
        
        # Today's interactions
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM interactions WHERE DATE(timestamp) = ?", (today,))
        today_count = cursor.fetchone()[0]
        print(f"Today's interactions: {today_count}")
        
        # Most active users
        cursor.execute("""
            SELECT user_id, COUNT(*) as count 
            FROM interactions 
            GROUP BY user_id 
            ORDER BY count DESC 
            LIMIT 5
        """)
        print("\n🏆 Top 5 most active users:")
        for user_id, count in cursor.fetchall():
            print(f"  {user_id}: {count} interactions")
    
    if 'admin_users' in tables:
        cursor.execute("SELECT username, created_at FROM admin_users")
        admins = cursor.fetchall()
        print(f"\n👑 Admin users: {len(admins)}")
        for username, created_at in admins:
            print(f"  {username} - created: {created_at}")
    
    conn.close()
    print("\n✅ Database closed.")

if __name__ == "__main__":
    view_database()
    input("\nPress Enter to exit...")