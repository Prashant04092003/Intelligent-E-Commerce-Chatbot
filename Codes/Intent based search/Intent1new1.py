"""
Streamlit Intent Recommender with enhanced UI
Uses precomputed embeddings - no API keys needed
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import requests
import ast
import re
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ------------------ CONFIG ------------------
DATA_PATH = "Codes/Intent based search/cleaned_flipkart_data.csv"
EMBED_PATH = "Codes/Intent based search/flipkart_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10

PRODUCT_NAME_COL = "product_name"
CATEGORY_COL = "main_category"
PRICE_COL = "discounted_price"
IMAGE_COL = "image"
DESC_COL = "description"
RATING_COL = "overall_rating"
CATEGORY_TREE_COL = "product_category_tree"

OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Similarity calibration defaults (can be overridden from sidebar)
SIM_LOW_BAND = 0.38
SIM_HIGH_BAND = 0.93
# Global boost multiplier (set from sidebar in main())
BOOST_MULT = 1.0

# ------------------ STYLING ------------------
def local_css():
    st.markdown("""
    <style>
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .product-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }
        .product-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1a237e;
            margin-bottom: 0.5rem;
        }
        .product-category {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .product-price {
            color: #2e7d32;
            font-size: 1.4rem;
            font-weight: 600;
            margin: 0.5rem 0;
        }
        .product-meta {
            display: flex;
            justify-content: space-between;
            margin-top: 0.5rem;
            padding-top: 0.5rem;
            border-top: 1px solid #eee;
        }
        .product-score {
            color: #1976d2;
            font-weight: 500;
        }
        .product-rating {
            color: #ff9800;
            font-weight: 500;
        }
        /* Improve tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 4px;
            gap: 4px;
            padding: 8px 16px;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------ DATA LOADING ------------------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    
    # Extract main category from category tree - with better error handling
    def parse_category(cat_str):
        try:
            if pd.isna(cat_str):
                return "Unknown"
            # Clean up the string and try to parse
            cleaned = cat_str.replace("=>", ",").replace("'", '"')
            cats = ast.literal_eval(cleaned)
            return cats[0] if isinstance(cats, list) and cats else "Unknown"
        except:
            return "Unknown"
    
    df[CATEGORY_COL] = df[CATEGORY_TREE_COL].apply(parse_category)
    
    # Convert price to numeric, handling any commas
    try:
        df[PRICE_COL] = df[PRICE_COL].astype(str)
        df[PRICE_COL] = pd.to_numeric(
            df[PRICE_COL].str.replace(',', ''), 
            errors='coerce'
        )
    except Exception as e:
        st.error(f"Error processing price column: {e}")
        df[PRICE_COL] = df[PRICE_COL].astype(float)
    
    # Fill missing values
    df[PRODUCT_NAME_COL] = df[PRODUCT_NAME_COL].fillna('')
    df[DESC_COL] = df[DESC_COL].fillna('')
    df[CATEGORY_COL] = df[CATEGORY_COL].fillna('Unknown')
    df[RATING_COL] = df[RATING_COL].fillna(0)
    
    return df.reset_index(drop=True)  # Ensure clean index

@st.cache_data(ttl=3600)
def load_embeddings(expected_rows=None):
    """Load and normalize embeddings from .npy file"""
    try:
        if not os.path.exists(EMBED_PATH):
            raise FileNotFoundError(f"Embeddings not found: {EMBED_PATH}")
        
        arr = np.load(EMBED_PATH)
        
        if len(arr.shape) != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape {arr.shape}")
        
        if expected_rows and arr.shape[0] != expected_rows:
            raise ValueError(f"Embeddings rows {arr.shape[0]} != data rows {expected_rows}")
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        arr = arr / norms
        return arr.astype(np.float32)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading embeddings: {str(e)}")

# ------------------ SEARCH INDEX ------------------
def build_index(embeddings):
    """Build search index - embeddings should already be normalized"""
    d = embeddings.shape[1]
    if FAISS_AVAILABLE:
        # FAISS IndexFlatIP expects normalized vectors (inner product = cosine similarity)
        index = faiss.IndexFlatIP(d)
        index.add(embeddings.astype(np.float32))
        return ('faiss', index)
    else:
        # Sklearn NearestNeighbors with cosine metric expects normalized vectors
        nn = NearestNeighbors(n_neighbors=TOP_K, metric='cosine', algorithm='brute')
        nn.fit(embeddings)
        return ('sklearn', nn)

# ------------------ QUERY PREPROCESSING ------------------
def extract_keywords(query):
    """Extract important keywords from query"""
    # Common stop words to filter out
    stop_words = {'i', 'need', 'want', 'looking', 'for', 'a', 'an', 'the', 'is', 'are', 'was', 'were', 
                  'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'should', 'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those',
                  'to', 'of', 'in', 'on', 'at', 'by', 'with', 'from', 'as', 'and', 'or', 'but'}
    
    words = query.lower().split()
    keywords = [w for w in words if len(w) > 2 and w not in stop_words]
    return keywords

def expand_query(query):
    """Expand query with synonyms and related terms"""
    # Category-specific expansions
    expansions = {
        'running': ['jogging', 'sport', 'athletic', 'sneakers', 'trainers'],
        'shoes': ['footwear', 'sneakers', 'trainers', 'boots'],
        'comfortable': ['comfort', 'cushioned', 'soft', 'ergonomic'],
        'formal': ['business', 'office', 'dress', 'professional'],
        'casual': ['everyday', 'relaxed', 'informal'],
        'gym': ['fitness', 'workout', 'exercise', 'training'],
        'trekking': ['hiking', 'outdoor', 'trail', 'adventure']
    }
    
    expanded = [query]
    keywords = extract_keywords(query)
    
    for keyword in keywords:
        for key, synonyms in expansions.items():
            if key in query.lower():
                for syn in synonyms[:2]:  # Add top 2 synonyms
                    expanded.append(query.replace(key, syn))
    
    return ' '.join(expanded[:3])  # Return expanded query with max 3 variations combined

def detect_intent_category(query):
    """Detect expected product category from query intent"""
    query_lower = query.lower()
    
    # Category mappings based on keywords
    category_keywords = {
        'Clothing': ['shirt', 'dress', 'pants', 'jeans', 'top', 't-shirt', 'apparel'],
        'Footwear': ['shoe', 'sneaker', 'boot', 'sandal', 'slipper', 'footwear'],
        'Electronics': ['phone', 'laptop', 'tablet', 'camera', 'headphone', 'charger', 'electronic'],
        'Sports & Fitness': ['running', 'gym', 'fitness', 'workout', 'exercise', 'sport', 'jogging', 'trekking'],
        'Home & Kitchen': ['kitchen', 'cookware', 'furniture', 'home', 'bedroom'],
        'Beauty': ['makeup', 'cosmetic', 'skincare', 'beauty', 'perfume']
    }
    
    detected_categories = []
    for category, keywords in category_keywords.items():
        if any(kw in query_lower for kw in keywords):
            detected_categories.append(category)
    
    return detected_categories

def calculate_keyword_match_score(product_text, query_keywords):
    """Calculate additional score boost based on keyword matching"""
    if not query_keywords:
        return 0.0
    
    product_lower = str(product_text).lower()
    matches = sum(1 for kw in query_keywords if kw in product_lower)
    
    # Boost score based on keyword matches
    match_ratio = matches / len(query_keywords) if query_keywords else 0
    return min(match_ratio * 0.15, 0.15)  # Up to 15% boost

def rescale_similarity_array(similarities, low_band=0.35, high_band=0.97):
    """Rescale similarities into a tighter band for better score calibration."""
    try:
        sims = np.array(similarities, dtype=float)
        if sims.size == 0:
            return sims
        s_min = float(np.min(sims))
        s_max = float(np.max(sims))
        if s_max <= s_min + 1e-8:
            return np.clip(np.full_like(sims, (low_band + high_band)/2.0), 0.0, 1.0)
        scaled = (sims - s_min) / (s_max - s_min)
        mapped = low_band + scaled * (high_band - low_band)
        return np.clip(mapped, 0.0, 1.0)
    except Exception:
        return np.array(similarities, dtype=float)

def preprocess_query(query):
    """Preprocess query to improve matching accuracy"""
    if not query or not isinstance(query, str):
        return ""
    # Remove extra whitespace and trim
    query = ' '.join(query.strip().split())
    return query

# ------------------ RECOMMENDATIONS ------------------
def recommend(query, model, index_obj, mode, df, embeddings, top_k=None, min_relevance=0.3):
    """Get product recommendations based on query with enhanced intent matching"""
    # Preprocess query
    query = preprocess_query(query)
    if not query:
        return []
    
    # Extract keywords and detect expected categories
    query_keywords = extract_keywords(query)
    expected_categories = detect_intent_category(query)
    
    # Use provided top_k or default TOP_K (search for more to account for filtering)
    # Search for 3-4x the requested amount to ensure enough results after filtering
    final_k = top_k if top_k else TOP_K
    k = max(final_k * 4, 50)  # Search for at least 4x the requested results or 50, whichever is larger
    
    # Encode query (optionally with expansion)
    expanded_query = expand_query(query)
    q_emb = model.encode([expanded_query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    
    # Ensure query embedding is properly normalized
    q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    q_emb = q_emb / q_norm
    
    if mode == 'faiss':
        # FAISS returns inner product (cosine similarity for normalized vectors)
        D, I = index_obj[1].search(q_emb, k)
        sims = D[0]
        idxs = I[0]
        # Clip + rescale for better calibration
        sims = rescale_similarity_array(np.clip(sims, 0, 1), low_band=SIM_LOW_BAND, high_band=SIM_HIGH_BAND)
    else:
        # Sklearn returns cosine distance (1 - cosine similarity)
        D, I = index_obj[1].kneighbors(q_emb, n_neighbors=k)
        sims = 1 - D[0]  # Convert distance to similarity
        idxs = I[0]
        # Clip + rescale for better calibration
        sims = rescale_similarity_array(np.clip(sims, 0, 1), low_band=SIM_LOW_BAND, high_band=SIM_HIGH_BAND)

    # Sort by similarity (descending) to ensure best matches first
    sorted_indices = np.argsort(sims)[::-1]
    sims = sims[sorted_indices]
    idxs = idxs[sorted_indices]

    results = []
    for s, i in zip(sims, idxs):
        try:
            idx_int = int(i)
            if idx_int >= len(df) or idx_int < 0:
                continue
            item = df.iloc[idx_int].copy()
            
            # Base semantic similarity score
            base_score = float(s)
            
            # Build product text for keyword matching
            product_text = ' '.join([
                str(item.get(PRODUCT_NAME_COL, '')),
                str(item.get(DESC_COL, '')),
                str(item.get(CATEGORY_COL, '')),
                str(item.get(CATEGORY_TREE_COL, ''))
            ]).lower()
            
            # Calculate keyword match boost
            keyword_boost = calculate_keyword_match_score(product_text, query_keywords)
            # Title-focused boost: direct query keyword matches in product name are stronger
            try:
                product_name_lower = str(item.get(PRODUCT_NAME_COL, '')).lower()
                name_matches = sum(1 for kw in query_keywords if kw in product_name_lower)
                name_boost = min(0.10, 0.035 * name_matches)
                keyword_boost = min(0.25, keyword_boost + name_boost)
            except Exception:
                pass
            
            # Category relevance boost/penalty
            category_boost = 0.0
            product_category = str(item.get(CATEGORY_COL, '')).lower()
            
            if expected_categories:
                # Check if product category matches expected categories
                category_match = any(exp_cat.lower() in product_category or 
                                   product_category in exp_cat.lower() 
                                   for exp_cat in expected_categories)
                
                if category_match:
                    category_boost = 0.09  # slightly reduced to avoid overfitting
                else:
                    # Smaller penalty for mismatched categories (don't be too aggressive)
                    category_penalty = -0.05  # 5% penalty for mismatch (reduced from 15%)
                    base_score = max(0, base_score + category_penalty)
            
            # Small quality bump for highly-rated products
            rating_raw = item.get(RATING_COL, 0)
            try:
                rating_num = float(pd.to_numeric(rating_raw, errors='coerce')) if rating_raw is not None else 0
            except Exception:
                rating_num = 0
            rating_boost = 0.0
            if rating_num >= 4.5:
                rating_boost = 0.05
            elif rating_num >= 4.2:
                rating_boost = 0.025

            # Apply boost multiplier and cap total boost to prevent saturation
            total_boost = (keyword_boost + category_boost + rating_boost) * BOOST_MULT
            total_boost = min(total_boost, 0.42)
            # Weight semantic similarity higher than boosts
            final_score = min(1.0, base_score * 0.86 + total_boost)
            
            # Apply relevance threshold - but be lenient if we need more results
            # Only filter very low scores if we have enough candidates
            if final_score < min_relevance:
                # If score is extremely low (< 0.2) or we have many results already, filter it out
                if final_score < 0.2 or len(results) >= final_k * 2:
                    continue
                # Otherwise, keep it even if below threshold (we'll sort and limit later)
            
            item['score'] = final_score
            item['base_score'] = base_score
            item['keyword_match'] = keyword_boost
            item['category_match'] = category_boost
            item['matched_keywords'] = [kw for kw in query_keywords if kw in product_text]
            
            # Store original DataFrame index for filtering purposes
            if hasattr(item, 'name'):
                item['_original_index'] = item.name
            else:
                item['_original_index'] = df.index[idx_int]
            results.append(item)
        except (IndexError, ValueError):
            continue
    
    # Re-sort by final score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Limit to requested top_k, but ensure we return at least what was requested
    # If we filtered too aggressively, return more results
    if len(results) < final_k:
        # If we have fewer results than requested due to filtering, 
        # try to return what we have (at least 1, but ideally more)
        return results
    return results[:final_k]

# Add these functions after other helper functions
def generate_local_ai_analysis(query, product_info, score, price, rating, category):
    """Generate AI-like analysis without requiring external API"""
    query_lower = query.lower()
    product_lower = product_info.lower()
    
    analysis = "### 🤖 Product Analysis\n\n"
    
    # 1. Relevance Score
    if score >= 0.7:
        relevance = "Highly Relevant (9-10/10)"
        relevance_desc = "This product closely matches your search criteria."
    elif score >= 0.5:
        relevance = "Relevant (6-8/10)"
        relevance_desc = "This product matches your needs reasonably well."
    else:
        relevance = "Somewhat Relevant (4-5/10)"
        relevance_desc = "This product has partial relevance to your search."
    
    analysis += f"**🎯 Relevance Score:** {relevance}\n"
    analysis += f"*{relevance_desc}*\n\n"
    
    # 2. Key Benefits
    analysis += "**✅ Key Benefits:**\n"
    benefits = []
    
    # Match keywords
    query_keywords = extract_keywords(query)
    matched_kw = [kw for kw in query_keywords if kw in product_lower]
    if matched_kw:
        benefits.append(f"Matches your search terms: {', '.join(matched_kw[:3])}")
    
    # Price analysis (add to benefits if reasonable)
    if price and price > 0:
        if price < 1000:
            benefits.append(f"Affordable pricing at ₹{price:,.2f}")
        elif price < 50000:
            benefits.append(f"Competitive pricing at ₹{price:,.2f}")
    
    # Rating analysis (ensure rating is numeric)
    try:
        rating_num = float(rating) if rating is not None else 0
        if rating_num > 0:
            if rating_num >= 4.0:
                benefits.append(f"High customer rating ({rating_num:.1f}/5) indicates quality")
            elif rating_num >= 3.0:
                benefits.append(f"Decent customer rating ({rating_num:.1f}/5)")
    except (ValueError, TypeError):
        pass  # Skip rating analysis if conversion fails
    
    # Category match
    if category and category != 'Unknown':
        benefits.append(f"From '{category}' category which aligns with your search")
    
    if benefits:
        for benefit in benefits[:3]:
            analysis += f"- {benefit}\n"
    else:
        analysis += "- Product matches your search intent\n"
    analysis += "\n"
    
    # 3. Considerations
    analysis += "**⚠️ Things to Consider:**\n"
    considerations = []
    
    if score < 0.5:
        considerations.append("Match score is moderate - may want to review product details carefully")
    
    # Check rating (handle both numeric and string types)
    try:
        rating_num = float(rating) if rating is not None else 0
        if rating_num == 0:
            considerations.append("Limited rating information available - consider checking reviews")
    except (ValueError, TypeError):
        considerations.append("Limited rating information available - consider checking reviews")
    
    if price and price > 50000:
        considerations.append("Premium pricing - ensure it fits your budget")
    
    if considerations:
        for cons in considerations:
            analysis += f"- {cons}\n"
    else:
        analysis += "- Review product specifications and customer reviews before purchasing\n"
    analysis += "\n"
    
    # 4. Recommendation
    analysis += "**💡 Recommendation:**\n"
    if score >= 0.7:
        analysis += "**Strong Match** - This product appears to be a good fit for your needs. Review the details and specifications to confirm.\n"
    elif score >= 0.5:
        analysis += "**Good Option** - Consider this product, but also explore other similar options to find the best match.\n"
    else:
        analysis += "**Review Needed** - This product may partially meet your needs. Consider refining your search for better matches.\n"
    
    return analysis

def get_mistral_recommendations(query, product_info, score=None, price=None, rating=None, category=None):
    """Get smarter AI recommendations using product context"""
    # Convert rating to numeric if provided
    rating_numeric = 0
    if rating is not None:
        try:
            rating_numeric = pd.to_numeric(rating, errors='coerce')
            if pd.isna(rating_numeric):
                rating_numeric = 0
            else:
                rating_numeric = float(rating_numeric)
        except (ValueError, TypeError):
            rating_numeric = 0
    
    # Convert price to numeric if provided
    price_numeric = 0
    if price is not None:
        try:
            price_numeric = pd.to_numeric(price, errors='coerce')
            if pd.isna(price_numeric):
                price_numeric = 0
            else:
                price_numeric = float(price_numeric)
        except (ValueError, TypeError):
            price_numeric = 0
    
    # Try Ollama first if available
    try:
        response = requests.post(
            OLLAMA_API,
            json={"model": OLLAMA_MODEL, "prompt": f"""As a shopping assistant, analyze this product for the user's needs:

User Query: "{query}"
Product Details: {product_info}

Provide a concise analysis:
1. Relevance Score (1-10): How well does this match their needs?
2. Key Benefits: What makes this a good choice?
3. Limitations: What should they watch out for?
4. Alternative Suggestion: What else might work better?

Keep each point brief but insightful.""", "stream": False},
            timeout=5
        )
        if response.status_code == 200:
            result = response.json().get('response', '')
            if result:
                return result
    except (requests.exceptions.RequestException, requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        pass
    except Exception:
        pass
    
    # Fallback to local analysis
    return generate_local_ai_analysis(query, product_info, score or 0.5, price_numeric, rating_numeric, category or 'Unknown')

def extract_first_image_url(image_str):
    try:
        if pd.isna(image_str) or not image_str:
            return None

        # Step 1: try JSON
        try:
            images = json.loads(image_str)
        except:
            # Step 2: try python literal
            try:
                images = ast.literal_eval(image_str)
            except:
                # Step 3: fallback → treat as plain string
                return None

        if isinstance(images, list) and len(images) > 0:
            url = str(images[0])

            # Fix http → https
            if url.startswith("http://"):
                url = url.replace("http://", "https://")

            return url

        return None

    except Exception:
        return None

def format_specifications(specs_str):
    """Parse various spec formats (JSON, Ruby hash-rocket, Python literal) and return DataFrame."""
    try:
        # If already a list/dict, normalize
        if isinstance(specs_str, (list, dict)):
            parsed = specs_str
        else:
            # Try JSON first
            try:
                parsed = json.loads(specs_str)
            except Exception:
                # Replace Ruby "=>" with ":" and try JSON
                cleaned = specs_str.replace("=>", ":").replace("nil", "null")
                try:
                    parsed = json.loads(cleaned)
                except Exception:
                    # Try Python literal (after converting => -> :)
                    cleaned_py = specs_str.replace("=>", ":").replace(": nil", ": None").replace(": null", ": None")
                    parsed = ast.literal_eval(cleaned_py)
        # parsed can be dict like {"product_specification": [...]}
        items = []
        if isinstance(parsed, dict):
            # common key name variants
            for possible in ("product_specification", "product_specifications", "specifications", "spec"):
                if possible in parsed:
                    parsed_value = parsed[possible]
                    break
            else:
                # maybe dict itself is list-like of key/value pairs
                parsed_value = parsed
            if isinstance(parsed_value, list):
                items = parsed_value
            elif isinstance(parsed_value, dict):
                # convert dict to list of key/value
                items = [{"key": k, "value": v} for k, v in parsed_value.items()]
            else:
                # not expected, fallback to single entry
                items = [{"key": str(k), "value": str(v)} for k, v in parsed.items()]
        elif isinstance(parsed, list):
            items = parsed
        else:
            # fallback: try regex to extract "key"=>"value" pairs
            matches = re.findall(r'["\']?([^"\']+?)["\']?\s*=>\s*["\']?([^"\']+?)["\']?(?:,|\])', specs_str)
            if matches:
                items = [{"key": m[0].strip(), "value": m[1].strip()} for m in matches]

        # Normalize to list of (key, value)
        rows = []
        for it in items:
            if isinstance(it, dict):
                k = it.get("key") or it.get("name") or next(iter(it.keys()), "")
                v = it.get("value") or it.get("val") or (it.get(k) if k in it else "")
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                k, v = it[0], it[1]
            else:
                continue
            rows.append((str(k).strip(), str(v).strip()))
        if not rows:
            return None
        # return pandas DataFrame
        df = pd.DataFrame(rows, columns=["Specification", "Value"])
        return df
    except Exception:
        return None

def display_product_card(col, result, query, index):
    """Enhanced product card with better UI and AI insights"""
    with col:
        with st.container():
            # Product Image - handle JSON array of URLs
            if IMAGE_COL in result:
                img_url = extract_first_image_url(result[IMAGE_COL])
                if img_url:
                    try:
                        st.image(img_url, width=200)
                    except:
                        st.image("https://via.placeholder.com/200?text=No+Image", width=200)
                else:
                    st.image("https://via.placeholder.com/200?text=No+Image", width=200)
            
            # Relevance indicator
            score = result.get('score', 0)
            relevance_badge = ""
            if score >= 0.7:
                relevance_badge = "🟢 Highly Relevant"
                badge_color = "#2e7d32"
            elif score >= 0.5:
                relevance_badge = "🟡 Relevant"
                badge_color = "#f57c00"
            else:
                relevance_badge = "🟠 Somewhat Relevant"
                badge_color = "#e64a19"
            
            # Main Product Info
            st.markdown(f"""
            <div class="product-card">
                <div class="product-title">{result[PRODUCT_NAME_COL]}</div>
                <div class="product-category">{result[CATEGORY_COL]}</div>
                <div class="product-price">₹{result[PRICE_COL]:,.2f}</div>
                <div style="background-color: {badge_color}20; padding: 4px 8px; border-radius: 4px; margin: 4px 0; font-size: 0.85rem;">
                    <strong>{relevance_badge}</strong> - Match Score: {result['score']:.3f}
                </div>
                <div class="product-meta">
                    <span class="product-score">Match: {result['score']:.3f}</span>
                    <span class="product-rating">Rating: {result[RATING_COL]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show why this product matched
            if 'matched_keywords' in result and result['matched_keywords']:
                with st.expander("🔍 Why this matches your search", expanded=False):
                    keywords_display = ', '.join(result['matched_keywords'])
                    st.caption(f"**Matched keywords:** {keywords_display}")
                    if 'category_match' in result and result['category_match'] > 0:
                        st.caption("✅ Category relevance detected")
                    if 'keyword_match' in result:
                        st.caption(f"**Keyword match boost:** +{result['keyword_match']*100:.1f}%")

            # Product Description & AI Analysis in tabs
            tab1, tab2 = st.tabs(["📝 Details", "🤖 AI Analysis"])
            
            with tab1:
                if DESC_COL in result and result[DESC_COL] and str(result[DESC_COL]).strip():
                    desc_text = str(result[DESC_COL])
                    st.markdown(f"**Product Description:**\n{desc_text[:300]}{'...' if len(desc_text) > 300 else ''}")
                else:
                    st.info("No description available")
                
                specs = result.get('product_specifications') or result.get('product_specification')
                if specs and str(specs).strip():
                    with st.expander("📋 Product Specifications", expanded=False):
                        spec_df = format_specifications(specs)
                        if spec_df is not None and not spec_df.empty:
                            # show compact table
                            st.dataframe(spec_df, use_container_width=True, hide_index=True)
                        else:
                            # fallback - show raw string prettified
                            try:
                                cleaned = str(specs).replace("=>", ":")
                                pretty = json.dumps(json.loads(cleaned), indent=2)
                                st.code(pretty)
                            except Exception:
                                st.text(str(specs)[:500])
            
            with tab2:
                with st.spinner("Analyzing product..."):
                    product_info = f"{result.get(PRODUCT_NAME_COL, 'Product')}"
                    if DESC_COL in result and result[DESC_COL]:
                        product_info += f" - {str(result[DESC_COL])[:200]}"
                    
                    # Get product details for analysis
                    product_score = result.get('score', 0.5)
                    product_price = result.get(PRICE_COL, 0)
                    # Convert rating to numeric, handling strings and NaN
                    product_rating_raw = result.get(RATING_COL, 0)
                    try:
                        product_rating = pd.to_numeric(product_rating_raw, errors='coerce')
                        if pd.isna(product_rating):
                            product_rating = 0
                        else:
                            product_rating = float(product_rating)
                    except (ValueError, TypeError):
                        product_rating = 0
                    product_category = result.get(CATEGORY_COL, 'Unknown')
                    
                    # Generate AI analysis (with fallback)
                    rec = get_mistral_recommendations(
                        query, product_info, 
                        score=product_score,
                        price=product_price,
                        rating=product_rating,
                        category=product_category
                    )
                    st.markdown(rec)

                    # Show additional context
                    st.markdown("---")
                    with st.expander("📊 Technical Details", expanded=False):
                        st.write(f"**Match Score Breakdown:**")
                        st.write(f"- Base Semantic Score: {result.get('base_score', product_score):.3f}")
                        if 'keyword_match' in result:
                            st.write(f"- Keyword Boost: +{result['keyword_match']*100:.1f}%")
                        if 'category_match' in result:
                            st.write(f"- Category Match: {'Yes' if result['category_match'] > 0 else 'No'}")
                        if 'matched_keywords' in result and result['matched_keywords']:
                            st.write(f"- Matched Keywords: {', '.join(result['matched_keywords'][:5])}")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    """Load and cache the sentence transformer model"""
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

@st.cache_data
def group_by_category(results):
    """Group results by main category"""
    categorized = {}
    for item in results:
        # Use the already extracted category column
        main_cat = item.get(CATEGORY_COL, "Unknown")
        if main_cat not in categorized:
            categorized[main_cat] = []
        categorized[main_cat].append(item)
    return categorized

def calculate_value_score(row):
    """Calculate value-for-money score based on price, rating, and match score"""
    score = row.get('score', 0) * 0.5  # 50% weight on match score
    rating = row.get(RATING_COL, 0)
    # Convert rating to numeric if needed
    try:
        rating_num = pd.to_numeric(rating, errors='coerce')
        if pd.notna(rating_num) and rating_num > 0:
            score += (float(rating_num) / 5.0) * 0.3  # 30% weight on rating
    except (ValueError, TypeError):
        pass  # Skip rating if conversion fails
    price = row.get(PRICE_COL, float('inf'))
    if price > 0 and price < float('inf'):
        # Lower price = higher value score (normalized)
        score += 0.2  # Base value score
    return min(score, 1.0)

def generate_shopping_insights(results, query):
    """Generate comprehensive shopping insights and visualizations"""
    if not results:
        return {}, []
    
    insights = {}
    figures = []
    
    try:
        # Convert results to DataFrame with proper type handling
        results_df = pd.DataFrame(results)
        
        # Ensure numeric columns are properly typed
        if PRICE_COL in results_df:
            results_df[PRICE_COL] = pd.to_numeric(results_df[PRICE_COL], errors='coerce')
        if RATING_COL in results_df:
            results_df[RATING_COL] = pd.to_numeric(results_df[RATING_COL], errors='coerce')
        if 'score' in results_df:
            results_df['score'] = pd.to_numeric(results_df['score'], errors='coerce')
        
        # Calculate value scores
        if not results_df.empty:
            try:
                results_df['value_score'] = results_df.apply(calculate_value_score, axis=1)
            except Exception:
                # Fallback if value score calculation fails
                results_df['value_score'] = results_df.get('score', 0)
        
        # 1. Price distribution (violin plot for better distribution view)
        if PRICE_COL in results_df and not results_df[PRICE_COL].isna().all():
            prices_clean = results_df[PRICE_COL].dropna()
            if len(prices_clean) > 0:
                fig_price = px.violin(
                    results_df,
        y=PRICE_COL,
                    title="📊 Price Distribution",
                    labels={PRICE_COL: "Price (₹)"},
                    box=True,
                    points="all"
                )
                fig_price.update_layout(height=400)
                figures.append(("Price Distribution", fig_price))
                
                # Price statistics
                insights['price_stats'] = {
                    'min': float(prices_clean.min()),
                    'max': float(prices_clean.max()),
                    'median': float(prices_clean.median()),
                    'mean': float(prices_clean.mean()),
                    'std': float(prices_clean.std()) if len(prices_clean) > 1 else 0
                }
        
        # 2. Enhanced Category distribution (with scores)
        categories = [r.get(CATEGORY_COL, 'Unknown') for r in results if CATEGORY_COL in r]
        if categories:
            cat_counts = Counter(categories)
            cat_data = pd.DataFrame({
                'Category': list(cat_counts.keys()),
                'Count': list(cat_counts.values())
            })
            
            # Calculate average scores per category
            cat_scores = {}
            for r in results:
                cat = r.get(CATEGORY_COL, 'Unknown')
                if cat not in cat_scores:
                    cat_scores[cat] = []
                cat_scores[cat].append(r.get('score', 0))
            
            cat_data['Avg_Score'] = [np.mean(cat_scores.get(cat, [0])) for cat in cat_data['Category']]
            
            fig_cats = px.bar(
                cat_data,
                x='Category',
                y='Count',
                color='Avg_Score',
                title="📦 Category Distribution (Colored by Avg Match Score)",
                labels={'Count': 'Number of Products', 'Avg_Score': 'Avg Match Score'},
                color_continuous_scale='Blues'
            )
            fig_cats.update_xaxes(tickangle=-45)
            fig_cats.update_layout(height=400)
            figures.append(("Category Distribution", fig_cats))
            
            insights['top_category'] = cat_data.loc[cat_data['Count'].idxmax(), 'Category']
        
        # 3. Rating distribution
        if RATING_COL in results_df and not results_df[RATING_COL].isna().all():
            ratings_clean = results_df[RATING_COL].dropna()
            if len(ratings_clean) > 0:
                fig_ratings = px.histogram(
                    results_df,
                    x=RATING_COL,
                    title="⭐ Ratings Distribution",
                    labels={RATING_COL: "Rating", 'count': 'Number of Products'},
                    nbins=10,
                    color_discrete_sequence=['#ff9800']
                )
                fig_ratings.update_layout(height=400)
                figures.append(("Ratings Distribution", fig_ratings))
                
                insights['rating_stats'] = {
                    'min': float(ratings_clean.min()),
                    'max': float(ratings_clean.max()),
                    'avg': float(ratings_clean.mean()),
                    'median': float(ratings_clean.median())
                }
        
        # 4. Price vs Rating scatter plot (if both available)
        if (PRICE_COL in results_df and RATING_COL in results_df and 
            not results_df[PRICE_COL].isna().all() and not results_df[RATING_COL].isna().all()):
            price_rating_df = results_df[[PRICE_COL, RATING_COL, 'score']].dropna()
            if len(price_rating_df) > 0:
                fig_scatter = px.scatter(
                    price_rating_df,
                    x=PRICE_COL,
                    y=RATING_COL,
                    size='score',
                    color='score',
                    title="💰 Price vs Rating (Size = Match Score)",
                    labels={
                        PRICE_COL: 'Price (₹)',
                        RATING_COL: 'Rating',
                        'score': 'Match Score'
                    },
                    color_continuous_scale='Viridis',
                    hover_data=['score']
                )
                fig_scatter.update_layout(height=400)
                figures.append(("Price vs Rating", fig_scatter))
        
        # 5. Match Score distribution
        if 'score' in results_df and not results_df['score'].isna().all():
            fig_scores = px.histogram(
                results_df,
                x='score',
                title="🎯 Match Score Distribution",
                labels={'score': 'Match Score', 'count': 'Number of Products'},
                nbins=15,
                color_discrete_sequence=['#1976d2']
            )
            fig_scores.update_layout(height=400)
            figures.append(("Match Score Distribution", fig_scores))
            
            insights['score_stats'] = {
                'min': float(results_df['score'].min()),
                'max': float(results_df['score'].max()),
                'avg': float(results_df['score'].mean()),
                'median': float(results_df['score'].median())
            }
        
        # 6. Value-for-Money analysis
        if 'value_score' in results_df and not results_df['value_score'].isna().all():
            # Ensure value_score is numeric and sortable
            value_scores_clean = pd.to_numeric(results_df['value_score'], errors='coerce')
            if not value_scores_clean.isna().all():
                # Sort by value_score and get top products
                results_df_sorted = results_df.copy()
                results_df_sorted['_sort_score'] = value_scores_clean
                top_value = results_df_sorted.nlargest(3, '_sort_score')
                
                insights['top_value_products'] = []
                for _, row in top_value.iterrows():
                    product_name = str(row.get(PRODUCT_NAME_COL, 'Product'))[:60]
                    insights['top_value_products'].append({
                        'name': product_name,
                        'value_score': round(float(row.get('value_score', 0)), 3),
                        'price': float(row.get(PRICE_COL, 0)) if pd.notna(row.get(PRICE_COL)) else 0,
                        'rating': float(row.get(RATING_COL, 0)) if pd.notna(row.get(RATING_COL)) else 0,
                        'match_score': round(float(row.get('score', 0)), 3)
                    })
                
                # Create visualization with top 10 (or fewer if less available)
                top_10_value = results_df_sorted.nlargest(min(10, len(results_df)), '_sort_score')
                if len(top_10_value) > 0:
                    # Truncate product names for display
                    top_10_value_display = top_10_value.copy()
                    top_10_value_display['display_name'] = top_10_value_display[PRODUCT_NAME_COL].astype(str).str[:40]
                    
                    fig_value = px.bar(
                        top_10_value_display,
                        x='value_score',
                        y='display_name',
                        orientation='h',
                        title="🏆 Top Value-for-Money Products",
                        labels={'value_score': 'Value Score', 'display_name': 'Product'},
                        color='value_score',
                        color_continuous_scale='Greens'
                    )
                    fig_value.update_layout(height=min(500, len(top_10_value) * 50), yaxis={'categoryorder': 'total ascending'})
                    figures.append(("Top Value Products", fig_value))
        
        # 7. Price Range breakdown
        if PRICE_COL in results_df and not results_df[PRICE_COL].isna().all():
            prices_clean = results_df[PRICE_COL].dropna()
            if len(prices_clean) > 0:
                q33 = float(prices_clean.quantile(0.33))
                q67 = float(prices_clean.quantile(0.67))
                min_p = float(prices_clean.min())
                max_p = float(prices_clean.max())
                
                price_ranges = [
                    (f'Budget (< ₹{q33:,.0f})', (min_p, q33)),
                    (f'Mid-range (₹{q33:,.0f} - ₹{q67:,.0f})', (q33, q67)),
                    (f'Premium (> ₹{q67:,.0f})', (q67, max_p))
                ]
                
                range_counts = []
                for label, (range_min, range_max) in price_ranges:
                    # For budget: include items up to q33
                    # For mid-range: include items from q33 to q67
                    # For premium: include items from q67 to max
                    if 'Budget' in label:
                        count = len(prices_clean[(prices_clean < range_max)])
                    elif 'Premium' in label:
                        count = len(prices_clean[(prices_clean >= range_min)])
                    else:  # Mid-range
                        count = len(prices_clean[(prices_clean >= range_min) & (prices_clean < range_max)])
                    range_counts.append({'Range': label, 'Count': count})
                
                fig_ranges = px.pie(
                    pd.DataFrame(range_counts),
                    values='Count',
                    names='Range',
                    title="💵 Price Range Distribution",
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                fig_ranges.update_layout(height=400)
                figures.append(("Price Ranges", fig_ranges))
                
                insights['price_ranges'] = range_counts
        
        # 8. Top matches by score
        if 'score' in results_df and not results_df['score'].isna().all():
            # Ensure score is numeric for sorting
            scores_clean = pd.to_numeric(results_df['score'], errors='coerce')
            if not scores_clean.isna().all():
                results_df_sorted = results_df.copy()
                results_df_sorted['_sort_score'] = scores_clean
                top_matches = results_df_sorted.nlargest(5, '_sort_score')
                
                insights['top_matches'] = []
                for _, row in top_matches.iterrows():
                    insights['top_matches'].append({
                        'name': str(row.get(PRODUCT_NAME_COL, 'Product'))[:60],
                        'score': round(float(row.get('score', 0)), 3),
                        'category': str(row.get(CATEGORY_COL, 'Unknown')),
                        'price': float(row.get(PRICE_COL, 0)) if pd.notna(row.get(PRICE_COL)) else 0
                    })
        
        # 9. Additional relevant insights
        if len(results) > 0:
            # Best match insight
            if 'score' in results_df and not results_df['score'].isna().all():
                max_score = results_df['score'].max()
                insights['best_match_score'] = float(max_score)
                
            # Price insights
            if PRICE_COL in results_df and not results_df[PRICE_COL].isna().all():
                prices_clean = results_df[PRICE_COL].dropna()
                if len(prices_clean) > 0:
                    insights['affordability'] = {
                        'cheapest': float(prices_clean.min()),
                        'most_expensive': float(prices_clean.max()),
                        'price_gap': float(prices_clean.max() - prices_clean.min())
                    }
            
            # Category diversity
            categories = [str(r.get(CATEGORY_COL, 'Unknown')) for r in results if CATEGORY_COL in r]
            if categories:
                unique_cats = len(set(categories))
                insights['category_diversity'] = {
                    'unique_categories': unique_cats,
                    'total_products': len(categories),
                    'is_diverse': unique_cats > 1
                }
        
        return insights, figures
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.warning(f"Could not generate all insights: {str(e)}")
        # Return basic insights even if some fail
        basic_insights = {}
        if results:
            basic_insights['total_results'] = len(results)
            if 'score' in results[0]:
                scores = [r.get('score', 0) for r in results if 'score' in r]
                if scores:
                    basic_insights['avg_score'] = float(np.mean(scores))
        return basic_insights, []

def get_shopping_summary(results, query):
    """Generate an AI shopping summary"""
    summary_prompt = f"""As a shopping assistant, analyze these search results for the query: "{query}"

Key points to address:
1. Price Range: What's the typical price range for these items?
2. Popular Categories: Which product categories are most relevant?
3. Best Deals: Identify 2-3 best value-for-money options
4. Shopping Tips: What should the shopper consider?

Keep it concise and helpful."""

    try:
        response = requests.post(
            OLLAMA_API,
            json={"model": OLLAMA_MODEL, "prompt": summary_prompt, "stream": False},
            timeout=8
        )
        if response.status_code == 200:
            return response.json().get('response', '')
    except Exception as e:
        return f"Summary unavailable: {str(e)}"
    
    return "Shopping summary unavailable"

# ------------------ MAIN APP ------------------
def main():
    st.set_page_config(
        page_title="Intent-Based Product Recommender",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    local_css()

    # Header
    col1, col2 = st.columns([2,1])
    with col1:
        st.title("🛍️ Smart Product Finder")
        st.markdown("_Tell me what you're looking for and I'll find the best matches_")
    
    # Load data and models
    try:
        with st.spinner("Loading data and embeddings..."):
            df = load_data()
            if df.empty:
                st.error("Dataset is empty!")
                st.stop()
            
            embeddings = load_embeddings(len(df))
            if embeddings.shape[0] == 0:
                st.error("Embeddings are empty!")
                st.stop()
            
            model = load_model()
            index = build_index(embeddings)
        
        st.success(f"✅ Loaded {len(df)} products with {embeddings.shape[1]}-dimensional embeddings")
        
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🔍 Search Settings")
    top_k = st.sidebar.slider("Number of results", 4, 20, 8)
    
    # Categories for filtering
    categories = sorted(df[CATEGORY_COL].unique().tolist())
    selected_cats = st.sidebar.multiselect(
        "Filter by category",
        categories,
        default=[]
    )
    
    # Price range slider (use PRICE_COL constant)
    price_min = float(df[PRICE_COL].min()) if not df[PRICE_COL].isna().all() else 0.0
    price_max = float(df[PRICE_COL].max()) if not df[PRICE_COL].isna().all() else 100000.0
    price_range = st.sidebar.slider(
        "Price range (₹)",
        price_min,
        price_max,
        (price_min, price_max)
    )

    # Main search interface
    query = st.text_input(
        "🔍 Describe what you're looking for...",
        value="I need comfortable running shoes for daily jogging",
        help="Example: 'I'm going to a wedding and need formal shoes' or 'Looking for a gaming laptop'"
    )

    # Relevance threshold slider
    min_relevance = st.sidebar.slider(
        "Minimum relevance threshold",
        0.0, 1.0, 0.3, 0.05,
        help="Filter out results below this relevance score"
    )

    # Similarity calibration control
    calib = st.sidebar.selectbox(
        "Similarity calibration",
        ["Balanced", "Conservative", "Aggressive"],
        index=0,
        help="Controls how strongly raw similarities are stretched into higher scores"
    )
    global SIM_LOW_BAND, SIM_HIGH_BAND, BOOST_MULT
    if calib == "Conservative":
        SIM_LOW_BAND, SIM_HIGH_BAND = 0.28, 0.90
    elif calib == "Aggressive":
        SIM_LOW_BAND, SIM_HIGH_BAND = 0.55, 0.995
    else:
        SIM_LOW_BAND, SIM_HIGH_BAND = 0.38, 0.93

    # Boost strength (to avoid overfitting when too high)
    boost_strength = st.sidebar.selectbox(
        "Boost strength",
        ["Low", "Medium", "High"],
        index=1,
        help="Controls how strongly keyword/title/category/rating boosts affect the score"
    )
    if boost_strength == "Low":
        BOOST_MULT = 0.6
    elif boost_strength == "High":
        BOOST_MULT = 1.3
    else:
        BOOST_MULT = 1.0
    
    # Enable smart filtering option
    enable_smart_filtering = st.sidebar.checkbox(
        "Enable smart category filtering",
        value=True,
        help="Automatically filter out products from unrelated categories based on your query"
    )

    if st.button("🔍 Search", type="primary"):
        if not query or not query.strip():
            st.warning("Please enter a search query")
            return
            
        # Show query analysis
        with st.expander("🔎 Query Analysis", expanded=False):
            keywords = extract_keywords(query)
            expected_cats = detect_intent_category(query)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Extracted Keywords:**")
                st.write(", ".join(keywords) if keywords else "None detected")
            with col2:
                st.write("**Expected Categories:**")
                st.write(", ".join(expected_cats) if expected_cats else "Could not determine")
            
        with st.spinner("Finding the best matches..."):
            # Get more results than requested to account for filtering
            # Search for at least 3x the requested amount to ensure we have enough after filtering
            search_k = max(top_k * 3, 30) if (selected_cats or price_range[0] != price_min or price_range[1] != price_max) else top_k * 2
            results = recommend(
                query, model, index, index[0], df, embeddings, 
                top_k=search_k, min_relevance=min_relevance if enable_smart_filtering else 0.0
            )
            
            # Additional smart filtering based on categories if enabled
            # Only filter if we have too many results (be less aggressive)
            if enable_smart_filtering and expected_cats and len(results) > top_k:
                expected_cats_lower = [c.lower() for c in expected_cats]
                filtered_results = []
                for r in results:
                    product_cat = str(r.get(CATEGORY_COL, '')).lower()
                    # Keep if category matches or if score is very high (might be valid despite category)
                    if any(exp_cat in product_cat or product_cat in exp_cat for exp_cat in expected_cats_lower) or r['score'] >= 0.6:
                        filtered_results.append(r)
                    elif r['score'] < 0.4:  # Only filter very low scores with category mismatch
                        continue
                    else:
                        filtered_results.append(r)  # Medium score, keep but mark
                # Only limit if we still have more than needed after filtering
                if len(filtered_results) > top_k:
                    results = filtered_results[:top_k]
                else:
                    results = filtered_results
            
            # Filter results
            if selected_cats:
                results = [r for r in results if r.get(CATEGORY_COL, 'Unknown') in selected_cats]
            
            # Filter by price (handle NaN prices)
            filtered_results = []
            for r in results:
                price = r.get(PRICE_COL, 0)
                if pd.isna(price):
                    # Include items with NaN price if we don't have enough results
                    if len(filtered_results) < top_k:
                        filtered_results.append(r)
                    continue
                if price_range[0] <= price <= price_range[1]:
                    filtered_results.append(r)
            
            # Ensure we have at least some results
            if len(filtered_results) < top_k and len(results) > len(filtered_results):
                # Add back some results that were filtered out to reach top_k
                for r in results:
                    if r not in filtered_results and len(filtered_results) < top_k:
                        filtered_results.append(r)
            
            results = filtered_results[:top_k]  # Limit to requested number
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📱 Products", "📊 Insights", "💡 Summary"])
            
            with tab1:
                if not results:
                    st.info("No products found matching your criteria")
                    return
                
                # Update the product card display in main() to include Mistral recommendations
                for i in range(0, len(results), 2):
                    col1, col2 = st.columns(2)
                    
                    if i < len(results):
                        display_product_card(col1, results[i], query, index)
                    if i+1 < len(results):
                        display_product_card(col2, results[i+1], query, index)
            
            with tab2:
                st.subheader("📊 Comprehensive Shopping Insights")
                
                # Generate insights
                insights, figures = generate_shopping_insights(results, query)
                
                # Display key statistics in metrics
                if insights:
                    st.markdown("### 📈 Key Statistics")
                    stats_cols = st.columns(4)
                    
                    with stats_cols[0]:
                        if 'score_stats' in insights and 'avg' in insights['score_stats']:
                            st.metric("Avg Match Score", f"{insights['score_stats']['avg']:.3f}")
                        elif 'avg_score' in insights:
                            st.metric("Avg Match Score", f"{insights['avg_score']:.3f}")
                        else:
                            st.metric("Total Results", insights.get('total_results', len(results)))
                    with stats_cols[1]:
                        if 'price_stats' in insights and 'mean' in insights['price_stats']:
                            st.metric("Avg Price", f"₹{insights['price_stats']['mean']:,.0f}")
                        elif 'affordability' in insights:
                            st.metric("Price Range", f"₹{insights['affordability']['cheapest']:,.0f} - ₹{insights['affordability']['most_expensive']:,.0f}")
                        else:
                            prices = [r.get(PRICE_COL, 0) for r in results if PRICE_COL in r and pd.notna(r.get(PRICE_COL))]
                            if prices:
                                st.metric("Avg Price", f"₹{np.mean(prices):,.0f}")
                    with stats_cols[2]:
                        if 'rating_stats' in insights and 'avg' in insights['rating_stats']:
                            st.metric("Avg Rating", f"{insights['rating_stats']['avg']:.2f}")
                        else:
                            ratings = []
                            for r in results:
                                if RATING_COL in r:
                                    rating_val = r.get(RATING_COL, 0)
                                    if pd.notna(rating_val):
                                        rating_num = pd.to_numeric(rating_val, errors='coerce')
                                        if pd.notna(rating_num) and rating_num > 0:
                                            ratings.append(float(rating_num))
                            if ratings:
                                st.metric("Avg Rating", f"{np.mean(ratings):.2f}")
                    with stats_cols[3]:
                        if 'top_category' in insights:
                            st.metric("Top Category", str(insights['top_category'])[:20])
                        elif 'category_diversity' in insights:
                            st.metric("Categories", insights['category_diversity']['unique_categories'])
                
                st.markdown("---")
                
                # Quick Insights Summary
                if insights:
                    st.markdown("### 💡 Quick Insights")
                    insight_cols = st.columns(3)
                    
                    with insight_cols[0]:
                        if 'best_match_score' in insights:
                            st.info(f"🎯 **Best Match:** {insights['best_match_score']:.3f}")
                        if 'affordability' in insights:
                            gap = insights['affordability']['price_gap']
                            if gap > 0:
                                st.info(f"💰 **Price Gap:** ₹{gap:,.0f}")
                    
                    with insight_cols[1]:
                        if 'category_diversity' in insights:
                            div_info = insights['category_diversity']
                            if div_info['is_diverse']:
                                st.warning(f"📦 **Multiple Categories:** {div_info['unique_categories']} categories found")
                            else:
                                st.success(f"📦 **Focused:** All products in same category")
                    
                    with insight_cols[2]:
                        if 'top_value_products' in insights and len(insights['top_value_products']) > 0:
                            best_value = insights['top_value_products'][0]
                            st.success(f"🏆 **Best Value:** {best_value['name'][:30]}... (Score: {best_value['value_score']:.3f})")
                
                st.markdown("---")
                
                # Display all visualizations
                if figures:
                    st.markdown("### 📉 Visualizations")
                    
                    # Create tabs for different visualization groups
                    viz_tabs = st.tabs(["Distribution", "Value Analysis", "Relationships", "Top Products"])
                    
                    with viz_tabs[0]:  # Distribution tab
                        dist_figs = [f for name, f in figures if any(x in name for x in 
                            ['Price Distribution', 'Ratings Distribution', 'Match Score Distribution', 'Category Distribution', 'Price Ranges'])]
                        
                        for i in range(0, len(dist_figs), 2):
                            cols = st.columns(2)
                            if i < len(dist_figs):
                                with cols[0]:
                                    st.plotly_chart(dist_figs[i], use_container_width=True)
                            if i + 1 < len(dist_figs):
                                with cols[1]:
                                    st.plotly_chart(dist_figs[i + 1], use_container_width=True)
                    
                    with viz_tabs[1]:  # Value Analysis tab
                        value_figs = [f for name, f in figures if 'Value' in name]
                        if value_figs:
                            for fig in value_figs:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Display top value products
                        if 'top_value_products' in insights:
                            st.markdown("#### 🏆 Best Value-for-Money Products")
                            for idx, product in enumerate(insights['top_value_products'], 1):
                                with st.expander(f"{idx}. {product['name'][:60]}..."):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Value Score", f"{product['value_score']:.3f}")
                                    with col2:
                                        price = product.get('price', 0)
                                        st.metric("Price", f"₹{price:,.2f}" if price > 0 else "N/A")
                                    with col3:
                                        st.metric("Match Score", f"{product['match_score']:.3f}")
                    
                    with viz_tabs[2]:  # Relationships tab
                        rel_figs = [f for name, f in figures if 'vs' in name or 'Price vs' in name]
                        if rel_figs:
                            for fig in rel_figs:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Relationship charts require both price and rating data")
                    
                    with viz_tabs[3]:  # Top Products tab
                        if 'top_matches' in insights:
                            st.markdown("#### 🎯 Top Matches by Score")
                            top_df = pd.DataFrame(insights['top_matches'])
                            st.dataframe(
                                top_df[['name', 'score', 'category', 'price']],
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'name': 'Product Name',
                                    'score': st.column_config.NumberColumn('Match Score', format="%.3f"),
                                    'category': 'Category',
                                    'price': st.column_config.NumberColumn('Price', format="₹%.2f")
                                }
                            )
                        
                        # Price statistics details
                        if 'price_stats' in insights:
                            st.markdown("#### 💰 Detailed Price Statistics")
                            ps = insights['price_stats']
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Minimum:** ₹{ps['min']:,.2f}")
                                st.write(f"**Maximum:** ₹{ps['max']:,.2f}")
                            with col2:
                                st.write(f"**Median:** ₹{ps['median']:,.2f}")
                                st.write(f"**Standard Deviation:** ₹{ps['std']:,.2f}")
                        
                        # Price range breakdown
                        if 'price_ranges' in insights:
                            st.markdown("#### 💵 Price Range Breakdown")
                            for range_info in insights['price_ranges']:
                                st.write(f"**{range_info['Range']}**: {range_info['Count']} products")
                else:
                    st.info("No insights data available. Try a different search query.")
                
                st.markdown("---")
                
                # Similar products recommendation
                st.subheader("🔍 You might also like")
                similar_cats = list(set(r.get(CATEGORY_COL, 'Unknown') for r in results))
                if similar_cats:
                    # Get indices of current results
                    result_indices = set()
                    for r in results:
                        # Use stored original index or Series name/index
                        if '_original_index' in r:
                            result_indices.add(r['_original_index'])
                        elif hasattr(r, 'name') and r.name is not None:
                            result_indices.add(r.name)
                        elif hasattr(r, 'index'):
                            # For Series, get the first index value
                            idx_vals = list(r.index)
                            if idx_vals:
                                result_indices.add(idx_vals[0])
                    
                    # Find similar products not in current results
                        similar_products = df[
                            (df[CATEGORY_COL].isin(similar_cats)) & 
                        (~df.index.isin(result_indices))
                    ]
                    
                    if len(similar_products) > 0:
                        similar_products = similar_products.sample(min(3, len(similar_products)))
                        similar_cols = st.columns(3)
                        for idx, (_, prod) in enumerate(similar_products.iterrows()):
                            with similar_cols[idx]:
                                price = prod.get(PRICE_COL, 0)
                                price_str = f"₹{price:,.2f}" if not pd.isna(price) else "Price unavailable"
                            st.markdown(f"""
                            <div class="product-card">
                                    <div class="product-title">{prod.get(PRODUCT_NAME_COL, 'Product')[:50]}</div>
                                    <div class="product-price">{price_str}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No additional similar products found")
                else:
                    st.info("Select products to see similar recommendations")
            
            with tab3:
                st.subheader("Shopping Summary")
                with st.spinner("Generating shopping insights..."):
                    summary = get_shopping_summary(results, query)
                    st.markdown(summary)
                    
                    # Price range recommendation
                    prices = [r.get(PRICE_COL, 0) for r in results if PRICE_COL in r and not pd.isna(r.get(PRICE_COL))]
                    if prices and len(prices) > 0:
                        median_price = np.median(prices)
                        mean_price = np.mean(prices)
                        st.info(f"""
                        💰 **Budget Guide**:
                        - Median Price: ₹{median_price:,.2f}
                        - Average Price: ₹{mean_price:,.2f}
                        - Price Range: ₹{min(prices):,.2f} - ₹{max(prices):,.2f}
                        - Budget Options: {sum(1 for p in prices if p < median_price)} products below median
                        - Premium Options: {sum(1 for p in prices if p > median_price)} products above median
                        """)
                    else:
                        st.info("Price information not available for these products")
                    
                    # Query refinement suggestions
                    st.markdown("---")
                    st.subheader("💡 Search Tips")
                    keywords = extract_keywords(query)
                    if len(keywords) < 3:
                        st.info("💬 **Tip:** Add more specific keywords (e.g., brand, size, color) for better results")
                    
                    # Show category distribution suggestion
                    result_categories = [r.get(CATEGORY_COL, 'Unknown') for r in results]
                    if result_categories:
                        unique_cats = set(result_categories)
                        if len(unique_cats) > 3:
                            top_cat = max(set(result_categories), key=result_categories.count)
                            st.info(f"💬 **Tip:** Most results are in '{top_cat}'. Try filtering by category for more focused results.")
                    
                    # Score distribution tip
                    if results:
                        avg_score = np.mean([r.get('score', 0) for r in results])
                        if avg_score < 0.5:
                            st.warning("⚠️ **Note:** Average match score is low. Try being more specific or using different keywords.")
                        elif avg_score >= 0.7:
                            st.success("✅ **Great!** Your query matches well with available products.")

    # Footer
    st.markdown("---")
    st.markdown("Enjoyed using the Smart Product Finder? Share your feedback!")

if __name__ == "__main__":
    main()
