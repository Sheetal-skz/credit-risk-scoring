"""
Custom styling and theme definitions for the Credit Risk Scoring Dashboard.
Provides a modern, high-trust fintech aesthetic with responsive cards,
crisp badges, and clean typography.
"""

def get_custom_css() -> str:
    return """
    <style>
        /* Global typography & spacing */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Top header accent bar */
        header[data-testid="stHeader"] {
            background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
            border-bottom: 1px solid #334155;
        }

        /* Hero banner */
        .hero-container {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F2847 100%);
            border-radius: 14px;
            padding: 26px 32px;
            margin-bottom: 24px;
            border: 1px solid #334155;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            color: #F8FAFC;
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0;
            background: linear-gradient(90deg, #FFFFFF 0%, #38BDF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            color: #94A3B8;
            margin-top: 8px;
            font-weight: 400;
            line-height: 1.5;
        }
        .hero-badge-strip {
            display: flex;
            gap: 12px;
            margin-top: 14px;
            flex-wrap: wrap;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            background: rgba(56, 189, 248, 0.12);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.25);
        }

        /* KPI / Metric Cards */
        .metric-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.2);
            border-color: #475569;
        }
        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.01em;
        }
        .metric-delta {
            font-size: 0.8rem;
            font-weight: 500;
            margin-top: 4px;
        }
        .delta-positive { color: #10B981; }
        .delta-neutral { color: #38BDF8; }
        .delta-caution { color: #F59E0B; }

        /* Decision Badges */
        .decision-badge-approved {
            display: inline-block;
            background: #064E3B;
            color: #34D399;
            font-weight: 800;
            font-size: 1.15rem;
            padding: 8px 18px;
            border-radius: 8px;
            border: 1px solid #059669;
            letter-spacing: 0.04em;
        }
        .decision-badge-review {
            display: inline-block;
            background: #78350F;
            color: #FBBF24;
            font-weight: 800;
            font-size: 1.15rem;
            padding: 8px 18px;
            border-radius: 8px;
            border: 1px solid #D97706;
            letter-spacing: 0.04em;
        }
        .decision-badge-declined {
            display: inline-block;
            background: #7F1D1D;
            color: #F87171;
            font-weight: 800;
            font-size: 1.15rem;
            padding: 8px 18px;
            border-radius: 8px;
            border: 1px solid #DC2626;
            letter-spacing: 0.04em;
        }

        /* Regulatory notice box */
        .notice-box {
            background: #0F172A;
            border: 1px solid #334155;
            border-left: 4px solid #38BDF8;
            border-radius: 6px;
            padding: 16px 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.84rem;
            color: #CBD5E1;
            white-space: pre-wrap;
            line-height: 1.5;
            margin-top: 12px;
        }

        /* Card container */
        .card-container {
            background: #1E293B;
            border-radius: 12px;
            border: 1px solid #334155;
            padding: 20px;
            margin-bottom: 20px;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 6px;
            padding: 0 16px;
            font-weight: 600;
            color: #94A3B8;
        }
        .stTabs [aria-selected="true"] {
            background-color: #334155 !important;
            color: #F8FAFC !important;
        }
    </style>
    """
