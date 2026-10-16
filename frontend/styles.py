"""
Design system and CSS styles for the Fintech Credit Risk Scoring Platform.
Matches the ultra-clean, modern, light sky-blue SaaS dashboard design language.
"""

def get_custom_css() -> str:
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Global Application Reset & Background */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background: linear-gradient(180deg, #EBF5FF 0%, #D8ECFD 100%) !important;
            color: #0F172A !important;
        }

        /* Hide Streamlit default header decoration */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1400px !important;
        }

        /* Top Navigation Bar */
        .top-navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0 20px 0;
            margin-bottom: 12px;
        }
        .top-navbar-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .app-logo {
            width: 38px;
            height: 38px;
            background: #0F172A;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #38BDF8;
            font-weight: 800;
            font-size: 1.2rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
        }
        .app-title-text {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.02em;
        }
        .top-navbar-right {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .search-pill {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 24px;
            padding: 8px 18px;
            font-size: 0.85rem;
            color: #64748B;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
        }
        .icon-circle-btn {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #475569;
            font-size: 0.95rem;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
        }
        .user-avatar-pill {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: linear-gradient(135deg, #0284C7 0%, #0EA5E9 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 0.85rem;
            border: 2px solid #FFFFFF;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.3);
        }

        /* Hero Portfolio Header */
        .hero-section {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 16px;
        }
        .hero-label {
            font-size: 0.9rem;
            font-weight: 500;
            color: #64748B;
            margin-bottom: 4px;
        }
        .hero-metric-group {
            display: flex;
            align-items: baseline;
            gap: 14px;
        }
        .hero-big-number {
            font-size: 2.85rem;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.03em;
            line-height: 1;
        }
        .hero-pill-badge {
            background: #DCFCE7;
            color: #15803D;
            font-size: 0.85rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 16px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .hero-subtext {
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 8px;
            font-weight: 500;
        }
        .hero-action-pills {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .date-range-badge {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 20px;
            padding: 8px 16px;
            font-size: 0.82rem;
            font-weight: 600;
            color: #334155;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
        }
        .quick-action-icon {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #475569;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
        }

        /* Clean White Cards */
        .white-card {
            background: #FFFFFF !important;
            border-radius: 20px !important;
            border: 1px solid #EDF2F7 !important;
            padding: 22px 24px !important;
            box-shadow: 0 4px 20px -2px rgba(100, 116, 139, 0.06) !important;
            margin-bottom: 20px !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .white-card:hover {
            box-shadow: 0 8px 25px -4px rgba(100, 116, 139, 0.1) !important;
        }

        /* 4 KPI Mini-Cards in First Row */
        .kpi-card {
            background: #FFFFFF;
            border-radius: 18px;
            border: 1px solid #EEF2F6;
            padding: 18px 20px;
            box-shadow: 0 2px 14px rgba(100, 116, 139, 0.05);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
            min-height: 140px;
            position: relative;
        }
        .kpi-icon-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .kpi-icon-disc {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
        }
        .kpi-icon-cyan { background: #E0F2FE; color: #0284C7; }
        .kpi-icon-teal { background: #CCFBF1; color: #0D9488; }
        .kpi-icon-pink { background: #FFE4E6; color: #E11D48; }
        .kpi-icon-blue { background: #DBEAFE; color: #2563EB; }

        .kpi-title {
            font-size: 0.82rem;
            color: #64748B;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .kpi-number {
            font-size: 1.75rem;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }
        .kpi-footer-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
        }
        .kpi-pill {
            font-size: 0.75rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            gap: 2px;
        }
        .kpi-pill-green { background: #DCFCE7; color: #15803D; }
        .kpi-pill-red { background: #FEE2E2; color: #B91C1C; }
        .kpi-arrow-link {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #94A3B8;
            font-size: 0.75rem;
            cursor: pointer;
        }

        /* Card Section Header with Arrow */
        .card-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .card-header-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.01em;
        }
        .card-header-more {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #64748B;
            font-size: 0.8rem;
            cursor: pointer;
        }

        /* List Items in Right Column Cards */
        .list-item-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #F1F5F9;
        }
        .list-item-row:last-child {
            border-bottom: none;
        }
        .list-item-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .list-item-icon {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: #F0F9FF;
            color: #0284C7;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
        }
        .list-item-title {
            font-size: 0.88rem;
            font-weight: 600;
            color: #0F172A;
        }
        .list-item-subtitle {
            font-size: 0.75rem;
            color: #94A3B8;
        }
        .list-item-right {
            text-align: right;
        }
        .list-item-value {
            font-size: 0.88rem;
            font-weight: 700;
            color: #0F172A;
        }
        .list-item-badge {
            font-size: 0.72rem;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 12px;
            display: inline-block;
        }
        .badge-design { background: #ECFDF5; color: #059669; border: 1px solid #A7F3D0; }
        .badge-marketing { background: #FFFBEB; color: #D97706; border: 1px solid #FDE68A; }
        .badge-software { background: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE; }
        .badge-danger { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }

        /* Decision Badges */
        .decision-badge-approved {
            display: inline-block;
            background: #ECFDF5;
            color: #059669;
            font-weight: 800;
            font-size: 1.1rem;
            padding: 8px 20px;
            border-radius: 12px;
            border: 1px solid #10B981;
            letter-spacing: 0.02em;
        }
        .decision-badge-review {
            display: inline-block;
            background: #FFFBEB;
            color: #D97706;
            font-weight: 800;
            font-size: 1.1rem;
            padding: 8px 20px;
            border-radius: 12px;
            border: 1px solid #F59E0B;
            letter-spacing: 0.02em;
        }
        .decision-badge-declined {
            display: inline-block;
            background: #FEF2F2;
            color: #DC2626;
            font-weight: 800;
            font-size: 1.1rem;
            padding: 8px 20px;
            border-radius: 12px;
            border: 1px solid #EF4444;
            letter-spacing: 0.02em;
        }

        /* Streamlit Input & Form Clean Overrides */
        .stSelectbox, .stNumberInput, .stSlider {
            color: #0F172A !important;
        }
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border-color: #E2E8F0 !important;
            border-radius: 12px !important;
            color: #0F172A !important;
        }
        input {
            background-color: #FFFFFF !important;
            border-color: #E2E8F0 !important;
            border-radius: 12px !important;
            color: #0F172A !important;
        }
        button[kind="primary"], .stButton > button {
            background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 10px 22px !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
            transition: all 0.2s ease !important;
        }
        button[kind="primary"]:hover, .stButton > button:hover {
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45) !important;
            transform: translateY(-1px) !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            box-shadow: 2px 0 16px rgba(0, 0, 0, 0.02) !important;
        }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: #0F172A !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: #334155 !important;
            font-weight: 500 !important;
        }

        /* Donut Chart Legend item */
        .legend-dot-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }
        .legend-dot-left {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #334155;
            font-weight: 500;
        }
        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .legend-amount {
            font-weight: 700;
            color: #0F172A;
        }

        /* Legal / Regulatory notice box */
        .notice-box {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #0284C7;
            border-radius: 12px;
            padding: 18px 22px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #334155;
            white-space: pre-wrap;
            line-height: 1.55;
            box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.02);
        }
    </style>
    """
