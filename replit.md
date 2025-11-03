# Menu Allergen Converter

## Overview

The Menu Allergen Converter is a Streamlit-based web application designed for school meal planning. It processes school menu data and automatically generates allergen-free versions by substituting ingredients that contain specified allergens (gluten, dairy, nuts, egg products, soy, fish). The application uses AI-powered suggestions via OpenAI's API to intelligently replace problematic ingredients while maintaining nutritional value and meal structure. Users can upload menu files, select allergens to exclude, review AI-suggested substitutions, and manage custom substitution rules.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology**: Streamlit web framework
- **Rationale**: Provides rapid development of data-centric applications with minimal frontend code
- **Structure**: Single-page application with sidebar configuration and main content area
- **State Management**: Uses Streamlit's session state to persist processed results, file hashes, and allergen selections across reruns
- **User Flow**: Upload menu → Select allergens → AI processes substitutions → Review/customize → Download modified menu

### Backend Architecture

**Core Components**:

1. **Menu Processing Pipeline** (`utils/menu_processor.py`)
   - Parses CSV/text menu files into structured pandas DataFrames
   - Extracts breakfast, lunch, and snack meals from formatted text
   - Maintains original menu structure for output formatting
   - **Design Pattern**: Parser/transformer pattern for data conversion

2. **AI Substitution Engine** (`utils/openai_service.py`)
   - Integrates with OpenAI API for intelligent ingredient substitution
   - Supports both single meal and batch processing for efficiency
   - Implements retry logic (3 attempts, 2-second delay) for API resilience
   - **Trade-off**: Uses external AI service (cost, latency) for better substitution quality vs rule-based approach

3. **Substitution Management** (`utils/substitutions.py`)
   - Combines custom user-defined rules with AI-generated suggestions
   - Provides CRUD operations for substitution rules
   - Prioritizes custom rules over AI suggestions for consistency
   - **Benefit**: Allows organizations to enforce specific dietary guidelines

4. **Confetti Animation** (`utils/confetti.py`)
   - Provides visual feedback using canvas-confetti JavaScript library
   - Enhances user experience upon successful processing
   - **Implementation**: Injects JavaScript via Streamlit's HTML component

### Data Storage

**Database**: PostgreSQL via SQLAlchemy ORM

**Schema Design**:

1. **SubstitutionRule Table**
   - Fields: id, allergen, original, replacement, created_at
   - Indexed on allergen for query performance
   - Stores user-defined substitution preferences

2. **Menu Table** (defined but usage not shown in current code)
   - Fields: id, name, content (JSON), created_at
   - Designed for storing historical menu data
   - JSON column allows flexible menu structure storage

**Database Configuration**:
- Connection pooling enabled with pre-ping health checks
- 1-hour connection recycling to prevent stale connections
- 10-second connection timeout for failure detection
- Application name set for database monitoring

**Alternatives Considered**:
- SQLite: Simpler but lacks concurrent access capabilities needed for multi-user scenarios
- NoSQL: More flexible but overkill for straightforward relational data

### External Dependencies

**Third-Party Services**:

1. **OpenAI API**
   - Purpose: Generate intelligent ingredient substitutions
   - Authentication: API key via environment variable (`OPENAI_API_KEY`)
   - Error Handling: Raises ValueError if API key not configured
   - Rate Limiting: Implements retry mechanism for transient failures

2. **Canvas Confetti CDN**
   - Purpose: Client-side animation library
   - URL: `https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js`
   - No authentication required (public CDN)

**Python Dependencies**:
- **streamlit**: Web application framework
- **pandas**: Data manipulation and CSV parsing
- **sqlalchemy**: ORM for database operations
- **openai**: Official OpenAI Python client

**Database**:
- **PostgreSQL**: Primary data store
- Connection string via `DATABASE_URL` environment variable
- Automatic URL transformation for SQLAlchemy compatibility (postgres:// → postgresql://)

**Environment Variables Required**:
- `DATABASE_URL`: PostgreSQL connection string
- `OPENAI_API_KEY`: OpenAI API authentication key

**Security Considerations**:
- Sensitive credentials stored in environment variables
- No hardcoded API keys or database credentials
- Database connection includes application identification for audit trails