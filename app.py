# app.py - COMPLETE VERSION WITH ALL ENDPOINTS
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, g
import json
import torch
from nltk_utils import tokenize, bag_of_words
from model import NeuralNet
import random
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from flask_cors import CORS
import requests
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'amreli-schemes-secret-key-2024'
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000"], supports_credentials=True)

# API Configuration
app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')
app.config['GOVT_SCHEMES_API'] = 'https://api.mygov.in/schemes'
app.config['AMRELI_PORTAL'] = 'https://amreli.gujarat.gov.in'
app.config['NATIONAL_PORTAL'] = 'https://india.gov.in'
DATABASE_PATH = 'schemes.db'

# Global variables for model and database
intents = None
model = None
all_words = None
tags = None
scheme_db = None

# -----------------------
# SQLite connection helpers (fix DB locked issues)
# -----------------------
def get_db():
    """
    Return a sqlite3.Connection stored in flask.g for the current request.
    Uses a reasonable timeout to reduce 'database is locked' errors.
    """
    db = getattr(g, '_database', None)
    if db is None:
        # timeout: how long to wait for locks; check_same_thread=False to allow usage from different threads
        db = g._database = sqlite3.connect(DATABASE_PATH, timeout=30, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(error):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# -----------------------
# Scheme Database Class - ENHANCED VERSION
# -----------------------
class SchemeDatabase:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with schemes table"""
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        c = conn.cursor()
        
        # Create schemes table
        c.execute('''
            CREATE TABLE IF NOT EXISTS schemes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag TEXT NOT NULL,
                category TEXT NOT NULL,
                scheme_name TEXT NOT NULL,
                description TEXT NOT NULL,
                eligibility TEXT,
                benefits TEXT,
                application_process TEXT,
                contact_info TEXT,
                website TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for better search performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_tag ON schemes(tag)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_category ON schemes(category)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_scheme_name ON schemes(scheme_name)')
        
        conn.commit()
        conn.close()
    
    def json_to_sql(self, json_file_path='intents.json'):
        """Convert JSON intents to SQL database with PROPER MAPPING"""
        try:
            # Load JSON data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                intents_data = json.load(f)
            
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            c = conn.cursor()
            
            # Clear existing data
            c.execute('DELETE FROM schemes')
            
            # COMPREHENSIVE category mapping from tags
            category_mapping = {
                'greeting': 'General',
                'schemecategories': 'General',
                'sarva_shiksha_abhiyan': 'Education',
                'midday_meal_scheme': 'Education',
                'pm_kisan': 'Agriculture',
                'kisan_credit_card': 'Agriculture',
                'ayushman_bharat': 'Healthcare',
                'jan_aushadhi': 'Healthcare',
                'pm_awas_yojana': 'Housing',
                'ujjwala_yojana': 'Women Empowerment',
                'mudra_yojana': 'Business & Startup',
                'sukanya_samriddhi': 'Women Empowerment',
                'atmanirbhar_bharat': 'Business & Startup',
                'digital_india': 'Digital Services',
                'goodbye': 'General',
                'thanks': 'General'
            }
            
            # Insert schemes with proper data extraction
            for intent in intents_data.get('intents', []):
                tag = intent.get('tag', '')
                category = category_mapping.get(tag, 'General')
                
                # Use the tag to create a proper scheme name
                name = ' '.join(word.capitalize() for word in tag.split('_'))
                
                # Extract description from responses
                description = ''
                if intent.get('responses'):
                    first_resp = intent['responses'][0]
                    if isinstance(first_resp, str):
                        # Clean the response for description
                        description = first_resp.replace('\n', ' ').replace('  ', ' ').strip()
                
                # Extract key information using pattern matching
                eligibility = self.extract_eligibility(intent)
                benefits = self.extract_benefits(intent)
                application_process = self.extract_application_process(intent)
                website = self.extract_website(intent)
                contact_info = self.extract_contact_info(intent)
                
                c.execute('''
                    INSERT INTO schemes (tag, category, scheme_name, description, eligibility, benefits, application_process, contact_info, website)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (tag, category, name, description, eligibility, benefits, application_process, contact_info, website))
            
            conn.commit()
            conn.close()
            print("✅ JSON to SQL import completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Error importing JSON to SQL: {e}")
            return False
    
    def extract_eligibility(self, intent):
        """Extract eligibility information from intent responses"""
        eligibility_keywords = ['eligibility', 'eligible', 'qualification', 'criteria', 'who can apply']
        return self.extract_info_by_keywords(intent, eligibility_keywords)
    
    def extract_benefits(self, intent):
        """Extract benefits information from intent responses"""
        benefits_keywords = ['benefits', 'benefit', 'advantages', 'features', 'what you get']
        return self.extract_info_by_keywords(intent, benefits_keywords)
    
    def extract_application_process(self, intent):
        """Extract application process information from intent responses"""
        application_keywords = ['application', 'apply', 'how to apply', 'process', 'procedure', 'steps']
        return self.extract_info_by_keywords(intent, application_keywords)
    
    def extract_website(self, intent):
        """Extract website information from intent responses"""
        for response in intent.get('responses', []):
            if isinstance(response, str):
                # Look for URLs
                import re
                urls = re.findall(r'https?://[^\s]+', response)
                if urls:
                    return urls[0]
        return ''
    
    def extract_contact_info(self, intent):
        """Extract contact information from intent responses"""
        contact_keywords = ['contact', 'phone', 'number', 'email', 'address', 'helpline']
        return self.extract_info_by_keywords(intent, contact_keywords)
    
    def extract_info_by_keywords(self, intent, keywords):
        """Extract information based on keywords"""
        for response in intent.get('responses', []):
            if isinstance(response, str):
                # Simple keyword matching
                for keyword in keywords:
                    if keyword.lower() in response.lower():
                        # Return the paragraph containing the keyword
                        lines = response.split('\n')
                        for line in lines:
                            if keyword.lower() in line.lower():
                                return line.strip()
                        return response[:200]  # Return first 200 chars if specific line not found
        return ''
    
    # ENHANCED SEARCH FUNCTION - More accurate and comprehensive
    def search_schemes(self, query, limit=50):
        """Enhanced search with better accuracy and ranking"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            q = f"%{query}%"
            
            # Enhanced search with multiple fields and better ranking
            c.execute('''
                SELECT id AS scheme_id, tag, category, scheme_name AS name, 
                       description, eligibility, benefits, application_process,
                       CASE 
                           WHEN lower(scheme_name) LIKE lower(?) THEN 4
                           WHEN lower(tag) LIKE lower(?) THEN 3
                           WHEN lower(category) LIKE lower(?) THEN 2
                           WHEN lower(description) LIKE lower(?) THEN 1
                           ELSE 0
                       END as relevance
                FROM schemes
                WHERE lower(scheme_name) LIKE lower(?) 
                   OR lower(description) LIKE lower(?) 
                   OR lower(tag) LIKE lower(?)
                   OR lower(category) LIKE lower(?)
                ORDER BY relevance DESC, scheme_name ASC
                LIMIT ?
            ''', (q, q, q, q, q, q, q, q, limit))
            
            rows = c.fetchall()
            conn.close()
            
            results = []
            for r in rows:
                if r['relevance'] > 0:  # Only include relevant results
                    results.append({
                        'id': r['scheme_id'],
                        'tag': r['tag'],
                        'category': r['category'],
                        'name': r['name'],
                        'description': r['description'],
                        'eligibility': r['eligibility'],
                        'benefits': r['benefits'],
                        'application_process': r['application_process'],
                        'relevance': r['relevance']
                    })
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_categories(self):
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        c = conn.cursor()
        c.execute('SELECT DISTINCT category FROM schemes')
        cats = [r[0] for r in c.fetchall()]
        conn.close()
        return cats
    
    def get_schemes_by_category(self, category_name):
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT id as scheme_id, tag, scheme_name as name, description, eligibility FROM schemes WHERE category = ?', (category_name,))
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def get_scheme_by_id(self, scheme_id):
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM schemes WHERE id = ?', (scheme_id,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def get_all_schemes(self, limit=100):
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT id as scheme_id, tag, category, scheme_name as name, description FROM schemes LIMIT ?', (limit,))
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]

# -----------------------
# Basic DB init / user/admin management
# -----------------------
def init_db():
    # This function creates supporting tables, and default admin users
    conn = sqlite3.connect(DATABASE_PATH, timeout=30, check_same_thread=False)
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME,
            login_count INTEGER DEFAULT 0
        )
    ''')
    
    # Create interactions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_message TEXT,
            bot_response TEXT,
            intent_tag TEXT,
            confidence REAL,
            action_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create admin users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create default admin users if they don't exist
    default_admins = [
        ('admin', 'admin123'),
        ('amreli_admin', 'amreli@2024')
    ]
    
    for username, password in default_admins:
        c.execute('SELECT * FROM admin_users WHERE username = ?', (username,))
        if not c.fetchone():
            password_hash = generate_password_hash(password)
            c.execute('INSERT INTO admin_users (username, password_hash) VALUES (?, ?)', (username, password_hash))
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")

# User management and login functions (use get_db() for request-time DB ops)
def register_user(email, name, password):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        if c.fetchone():
            return False, "Email already registered"
        password_hash = generate_password_hash(password)
        c.execute('INSERT INTO users (email, name, password_hash, last_login, login_count) VALUES (?, ?, ?, CURRENT_TIMESTAMP, 1)',
                 (email, name, password_hash))
        conn.commit()
        user_id = c.lastrowid
        log_interaction(str(user_id), 'USER_REGISTER', f'New user: {email}')
        return True, user_id
    except Exception as e:
        return False, str(e)

def login_user(email, password):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        if user and check_password_hash(user['password_hash'], password):
            c.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP, login_count = login_count + 1 WHERE id = ?', (user['id'],))
            conn.commit()
            log_interaction(str(user['id']), 'USER_LOGIN', 'Successful login')
            return True, user['id'], user['name']
        else:
            log_interaction('unknown', 'LOGIN_FAILED', f'Failed login: {email}')
            return False, None, None
    except Exception as e:
        print(f"Login error: {e}")
        return False, None, None

def get_user_stats():
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) as total FROM users')
        total_users = c.fetchone()['total']
        today = datetime.now().strftime('%Y-%m-%d')
        c.execute('SELECT COUNT(*) as new_today FROM users WHERE DATE(created_at) = ?', (today,))
        new_today = c.fetchone()['new_today']
        c.execute('SELECT COUNT(DISTINCT user_id) as active_today FROM interactions WHERE DATE(timestamp) = ?', (today,))
        active_today = c.fetchone()['active_today']
        return {'total_users': total_users, 'new_today': new_today, 'active_today': active_today}
    except Exception as e:
        print(f"User stats error: {e}")
        return {'total_users': 0, 'new_today': 0, 'active_today': 0}

# -----------------------
# AI Model & NLP helpers - UPDATED TO ALWAYS USE DATABASE
# -----------------------
def load_model():
    global intents, model, all_words, tags
    try:
        with open('intents.json', 'r', encoding='utf-8') as json_data:
            intents = json.load(json_data)
        
        # Try to load the improved model first, fall back to data.pth
        model_file = "data_improved.pth" if os.path.exists("data_improved.pth") else "data.pth"
        
        if not os.path.exists(model_file):
            print(f"❌ Model file '{model_file}' not found. Please run train.py first.")
            return False
            
        data = torch.load(model_file, map_location=torch.device('cpu'))
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data["all_words"]
        tags = data["tags"]
        model = NeuralNet(input_size, hidden_size, output_size)
        model.load_state_dict(data["model_state"])
        model.eval()
        print(f"✅ AI Model loaded successfully from {model_file}!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# SIMPLIFIED NLP RESPONSE - ALWAYS USE DATABASE SEARCH
def get_nlp_response(user_message):
    """Always return None to force database search for accurate responses"""
    return None

# -----------------------
# Logging
# -----------------------
def log_interaction(user_id, action_type, details="", bot_response=""):
    try:
        conn = get_db()
        c = conn.cursor()
        user_message = details
        intent_tag = ""
        confidence = 0.0
        if 'User message:' in details:
            user_message = details.replace('User message: ', '')
        c.execute('''
            INSERT INTO interactions (user_id, user_message, bot_response, intent_tag, confidence, action_type) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, user_message, bot_response, intent_tag, confidence, action_type))
        conn.commit()
    except Exception as e:
        print(f"Error logging interaction: {e}")

# -----------------------
# Routes - COMPLETE WITH ALL ENDPOINTS
# -----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin_login_page():
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect('/admin')
    return render_template('admin_dashboard.html')

# Serve the main chatbot pages
@app.route('/chatbot')
def chatbot_main():
    return render_template('index.html')

@app.route('/voice-assistant')
def voice_assistant():
    return render_template('index.html')

@app.route('/search-schemes')
def search_schemes_page():
    return render_template('index.html')

@app.route('/multilingual')
def multilingual():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Amreli Schemes Chatbot",
        "model_loaded": model is not None,
        "scheme_db_loaded": scheme_db is not None,
        "openai_configured": bool(app.config['OPENAI_API_KEY'] and app.config['OPENAI_API_KEY'] != '')
    })

# User endpoints
@app.route('/user/register', methods=['POST'])
def user_register():
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        name = data.get('name', '').strip()
        password = data.get('password', '').strip()
        if not all([email, name, password]):
            return jsonify({"success": False, "message": "All fields are required"})
        success, result = register_user(email, name, password)
        if success:
            return jsonify({"success": True, "user_id": result, "message": "Registration successful"})
        else:
            return jsonify({"success": False, "message": result})
    except Exception as e:
        return jsonify({"success": False, "message": "Registration failed"}), 500

@app.route('/user/login', methods=['POST'])
def user_login():
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        success, user_id, name = login_user(email, password)
        if success:
            return jsonify({"success": True, "user_id": user_id, "name": name, "message": "Login successful"})
        else:
            return jsonify({"success": False, "message": "Invalid email or password"})
    except Exception as e:
        return jsonify({"success": False, "message": "Login failed"}), 500

# UPDATED CHAT ENDPOINT - ALWAYS USE DATABASE
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', '')
        
        if not user_message:
            return jsonify({"response": "Please enter a message."})

        # ALWAYS search database for accurate responses
        db_results = scheme_db.search_schemes(user_message)
        
        if db_results and len(db_results) > 0:
            # Format response from database results
            response = format_database_response(db_results, user_message)
            response_source = "database"
        else:
            # Enhanced fallback that still provides useful information
            response = get_enhanced_fallback_response(user_message)
            response_source = "fallback"

        # Log interaction
        log_interaction(user_id or 'anonymous', 'CHAT', f'User message: {user_message}', bot_response=response)
        return jsonify({"response": response, "source": response_source})
        
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        return jsonify({"response": "I'm having trouble processing your request. Please try again."}), 500

def format_database_response(results, user_message):
    """Format database results into a user-friendly response"""
    if len(results) == 1:
        # Single result - provide detailed information
        scheme = results[0]
        response = f"**{scheme['name']}**\n\n"
        response += f"{scheme['description']}\n\n"
        
        if scheme.get('eligibility'):
            response += f"**Eligibility:** {scheme['eligibility']}\n\n"
        
        if scheme.get('benefits'):
            response += f"**Benefits:** {scheme['benefits']}\n\n"
            
        if scheme.get('application_process'):
            response += f"**How to Apply:** {scheme['application_process']}\n\n"
            
        response += f"**Category:** {scheme['category']}"
        
    else:
        # Multiple results - provide summary
        response = f"🔍 I found {len(results)} schemes matching '{user_message}':\n\n"
        
        for i, scheme in enumerate(results[:5], 1):  # Show top 5 results
            response += f"**{i}. {scheme['name']}** ({scheme['category']})\n"
            
            # Truncate description if too long
            desc = scheme['description']
            if len(desc) > 150:
                desc = desc[:150] + "..."
            response += f"   {desc}\n"
            
            if scheme.get('eligibility'):
                elig = scheme['eligibility']
                if len(elig) > 100:
                    elig = elig[:100] + "..."
                response += f"   *Eligibility:* {elig}\n"
                
            response += "\n"
        
        if len(results) > 5:
            response += f"💡 *Showing top 5 results. {len(results) - 5} more schemes available.*\n\n"
            
        response += "**💡 Tip:** Ask about any specific scheme for more detailed information!"
    
    return response

def get_enhanced_fallback_response(query):
    """Provide helpful fallback response with database information"""
    query_lower = query.lower()
    
    # Get all categories and some schemes to suggest
    categories = scheme_db.get_categories()
    all_schemes = scheme_db.get_all_schemes(limit=10)
    
    # Check if query matches any category
    matched_category = None
    for category in categories:
        if category.lower() in query_lower:
            matched_category = category
            break
    
    if matched_category:
        schemes_in_category = scheme_db.get_schemes_by_category(matched_category)
        response = f"**{matched_category} Schemes**\n\n"
        response += f"I found {len(schemes_in_category)} schemes in this category:\n\n"
        
        for i, scheme in enumerate(schemes_in_category[:5], 1):
            response += f"• **{scheme['name']}**\n"
        
        response += f"\n💡 Ask about any specific scheme for details!"
        
    else:
        # General fallback with available categories and schemes
        response = f"🔍 I couldn't find exact matches for '{query}'. "
        response += "Here are some available scheme categories:\n\n"
        
        for category in categories[:8]:  # Show first 8 categories
            response += f"• {category}\n"
        
        response += f"\n**Popular Schemes:**\n"
        for scheme in all_schemes[:5]:
            response += f"• {scheme['name']}\n"
            
        response += "\n💡 **Try these examples:**\n"
        response += "• 'Tell me about education schemes'\n"
        response += "• 'What agriculture programs are available?'\n"
        response += "• 'Show me healthcare benefits'\n"
        response += "• 'List housing schemes'"
    
    return response

# Scheme search endpoints - ADD ALL SEARCH ENDPOINTS
@app.route('/search_schemes', methods=['POST'])
def search_schemes():
    """Legacy endpoint for search_schemes"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        user_id = data.get('user_id', '')
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
            
        # Use the existing search functionality
        results = scheme_db.search_schemes(query, 50)
        
        # Log the search interaction
        log_interaction(user_id or 'anonymous', 'SEARCH', f'Search query: {query}')
        
        return jsonify({"results": results, "query": query, "total": len(results)})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search_schemes', methods=['POST'])
def api_search_schemes():
    """API endpoint for scheme search"""
    return search_schemes()

@app.route('/api/schemes/search', methods=['POST'])
def api_schemes_search():
    """Alternative search endpoint"""
    return search_schemes()

@app.route('/api/categories', methods=['GET'])
def get_categories():
    try:
        categories = scheme_db.get_categories()
        return jsonify({"categories": categories})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/schemes/category/<category_name>', methods=['GET'])
def get_schemes_by_category(category_name):
    try:
        schemes = scheme_db.get_schemes_by_category(category_name)
        return jsonify({"schemes": schemes, "category": category_name, "total": len(schemes)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/schemes/<int:scheme_id>', methods=['GET'])
def get_scheme_details(scheme_id):
    try:
        scheme = scheme_db.get_scheme_by_id(scheme_id)
        if scheme:
            return jsonify({"scheme": scheme})
        else:
            return jsonify({"error": "Scheme not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/schemes', methods=['GET'])
def get_all_schemes():
    try:
        limit = request.args.get('limit', 100, type=int)
        schemes = scheme_db.get_all_schemes(limit=limit)
        return jsonify({"schemes": schemes, "total": len(schemes)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Additional API endpoints
@app.route('/api/health', methods=['GET'])
def api_health():
    return health_check()

# Admin endpoints
@app.route('/admin/login', methods=['POST'])
def admin_login():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM admin_users WHERE username = ?', (username,))
        admin_user = c.fetchone()
        if admin_user and check_password_hash(admin_user['password_hash'], password):
            session['admin_logged_in'] = True
            session['admin_username'] = username
            log_interaction(username, 'ADMIN_LOGIN', 'Successful admin login')
            return jsonify({"success": True, "message": "Login successful"})
        else:
            log_interaction(username, 'ADMIN_LOGIN_FAILED', 'Failed login attempt')
            return jsonify({"success": False, "message": "Invalid username or password"}), 401
    except Exception as e:
        log_interaction('system', 'ERROR', f'Admin login error: {str(e)}')
        return jsonify({"success": False, "message": "Login failed"}), 500

@app.route('/admin/logout')
def admin_logout():
    username = session.get('admin_username', 'unknown')
    session.clear()
    log_interaction(username, 'ADMIN_LOGOUT', 'Admin logged out')
    return redirect('/admin')

@app.route('/admin/api/dashboard_data')
def admin_dashboard_data():
    if not session.get('admin_logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    try:
        conn = get_db()
        c = conn.cursor()

        # Get basic stats
        c.execute('SELECT COUNT(*) as total FROM interactions')
        total_interactions = c.fetchone()['total']
        
        c.execute('SELECT COUNT(DISTINCT user_id) as unique_users FROM interactions')
        unique_users = c.fetchone()['unique_users']
        
        today = datetime.now().strftime('%Y-%m-%d')
        c.execute('SELECT COUNT(*) as today_interactions FROM interactions WHERE DATE(timestamp) = ?', (today,))
        today_interactions = c.fetchone()['today_interactions']

        # Get recent interactions
        c.execute('''
            SELECT user_id, user_message, bot_response, timestamp, action_type 
            FROM interactions 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''')
        interactions = [dict(row) for row in c.fetchall()]

        # Get user stats
        user_stats = get_user_stats()
        
        # Get scheme stats
        scheme_stats = {
            'total_schemes': len(scheme_db.get_all_schemes(limit=1000000)),
            'total_categories': len(scheme_db.get_categories())
        }

        return jsonify({
            "total_interactions": total_interactions,
            "unique_users": unique_users,
            "today_interactions": today_interactions,
            "recent_interactions": interactions[:20],  # Last 20 interactions
            "user_stats": user_stats,
            "scheme_stats": scheme_stats
        })
    except Exception as e:
        print("admin_dashboard_data error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/admin/api/user_stats', methods=['GET'])
def admin_user_stats():
    if not session.get('admin_logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    try:
        user_stats = get_user_stats()
        return jsonify(user_stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin/api/scheme_stats', methods=['GET'])
def admin_scheme_stats():
    if not session.get('admin_logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    try:
        scheme_stats = {
            'total_schemes': len(scheme_db.get_all_schemes(limit=1000000)),
            'total_categories': len(scheme_db.get_categories())
        }
        return jsonify(scheme_stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# App initialization
# -----------------------
def initialize_app():
    print("🚀 Initializing Government Schemes Chatbot with Database-First Approach")
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("✅ Created templates directory")
    
    # Move HTML files to templates directory
    import shutil
    html_files = ['index.html', 'admin_login.html', 'admin_dashboard.html']
    for html_file in html_files:
        if os.path.exists(html_file) and not os.path.exists(f'templates/{html_file}'):
            shutil.copy(html_file, 'templates/')
            print(f"✅ Moved {html_file} to templates directory")
    
    # Initialize DB and scheme DB
    init_db()
    global scheme_db
    scheme_db = SchemeDatabase()
    
    # Force re-import of JSON data to ensure database is populated
    success = scheme_db.json_to_sql()
    if success:
        print("✅ Scheme database initialized and populated successfully!")
        
        # Show database statistics
        categories = scheme_db.get_categories()
        schemes = scheme_db.get_all_schemes()
        print(f"📊 Database contains {len(schemes)} schemes across {len(categories)} categories")
        print("📋 Categories:", ", ".join(categories))
    else:
        print("❌ Failed to initialize scheme database!")
    
    # Load AI model (optional now since we always use database)
    if load_model():
        print("✅ Local AI Model loaded successfully!")
    else:
        print("ℹ️ Local AI Model not available. Running in database-only mode.")
    
    print("📊 Admin Dashboard: http://localhost:5000/admin")
    print("🤖 Chatbot: http://localhost:5000/")
    print("🔧 Health Check: http://localhost:5000/health")

# Initialize when module is imported
initialize_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')