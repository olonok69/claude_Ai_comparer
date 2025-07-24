# Personal AGENDAS

A comprehensive data processing and analytics pipeline for veterinary conference data, designed to process registration, attendance, and session information to generate personalized recommendations and insights.

## Overview

This application processes data from veterinary conferences (BVA - British Veterinary Association and LVA - London Vet Show) to:
- Track visitor registration and demographics
- Analyze session attendance patterns
- Generate personalized session recommendations
- Create knowledge graphs in Neo4j for advanced analytics
- Provide insights into visitor behavior and preferences

## Features

### Data Processing
- **Registration Processing**: Handles visitor registration data, identifies returning visitors, and processes demographic information
- **Scan Processing**: Processes session attendance scan data and matches it with visitor profiles
- **Session Processing**: Manages session information, extracts stream categories, and generates AI-powered stream descriptions

### Neo4j Integration
- **Graph Database**: Creates a comprehensive knowledge graph with visitors, sessions, and streams
- **Relationship Mapping**: Establishes connections based on:
  - Job roles to relevant streams
  - Practice specializations to appropriate content
  - Historical attendance patterns
  - Visitor similarity metrics

### AI-Powered Features
- **Stream Descriptions**: Uses LLMs to generate descriptive summaries of session categories
- **Session Embeddings**: Creates semantic embeddings for content-based recommendations
- **Smart Recommendations**: Generates personalized session recommendations using:
  - Historical attendance data
  - Visitor similarity analysis
  - Business rules filtering
  - Optional LLM-based filtering

## Architecture

```
PA/app/
├── main.py                              # Main orchestrator
├── pipeline.py                          # Pipeline coordinator
├── registration_processor.py            # Registration data processing
├── scan_processor.py                    # Scan data processing
├── session_processor.py                 # Session data processing
├── neo4j_visitor_processor.py          # Neo4j visitor node creation
├── neo4j_session_processor.py          # Neo4j session node creation
├── neo4j_job_stream_processor.py       # Job role to stream relationships
├── neo4j_specialization_stream_processor.py  # Specialization relationships
├── neo4j_visitor_relationship_processor.py   # Visitor relationships
├── session_embedding_processor.py       # Session embedding generation
├── session_recommendation_processor.py  # Recommendation engine
├── utils/                              # Utility modules
│   ├── config_utils.py                 # Configuration management
│   ├── data_utils.py                   # Data manipulation utilities
│   ├── logging_utils.py                # Logging configuration
│   ├── neo4j_utils.py                  # Neo4j connection utilities
│   └── summary_utils.py                # Summary statistics generation
└── config/
    └── config.yaml                     # Main configuration file
```

## Prerequisites

- Python 3.8+
- Neo4j Database (4.4+)
- OpenAI API or Azure OpenAI credentials (for AI features)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PA/app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the `keys/` directory with:
```env
# Neo4j credentials
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# OpenAI or Azure OpenAI credentials
OPENAI_API_KEY=your-api-key
# OR for Azure:
AZURE_API_KEY=your-azure-key
AZURE_ENDPOINT=your-azure-endpoint
AZURE_DEPLOYMENT=your-deployment-name
AZURE_API_VERSION=2024-02-15-preview
```

4. Configure the application:
Edit `config/config.yaml` to set your data paths and processing options.

## Usage

### Full Pipeline Execution
Run the complete data processing pipeline:
```bash
python main.py
```

### Selective Processing
Run specific steps only:
```bash
# Run only registration and scan processing (steps 1 and 2)
python main.py --only-steps 1,2

# Skip Neo4j upload
python main.py --skip-neo4j

# Recreate all Neo4j nodes (even if they exist)
python main.py --recreate-all
```

### Standalone Processors
Run individual components:
```bash
# Generate session embeddings only
python run_embedding.py

# Generate recommendations only
python run_recommendations.py --min-score 0.3 --max-recommendations 10
```

## Data Flow

1. **Registration Processing**
   - Loads registration JSON files
   - Identifies returning visitors
   - Processes demographic data
   - Exports cleaned CSV files

2. **Scan Processing**
   - Loads session attendance scan data
   - Matches scans with visitor profiles
   - Enriches with demographic information

3. **Session Processing**
   - Processes session information
   - Extracts and categorizes streams
   - Generates AI descriptions for streams

4. **Neo4j Upload**
   - Creates visitor nodes (current and past years)
   - Creates session nodes with embeddings
   - Establishes relationships based on:
     - Job roles
     - Practice specializations
     - Historical attendance

5. **Recommendation Generation**
   - Analyzes visitor profiles
   - Finds similar visitors
   - Generates personalized recommendations
   - Applies business rules filtering

## Configuration

The `config/config.yaml` file controls all aspects of the pipeline:

```yaml
# Input file paths
input_files:
  bva_registration: "path/to/bva_registration.json"
  lvs_registration: "path/to/lvs_registration.json"
  # ... more input files

# Output directory
output_dir: "data/output"

# Pipeline control
pipeline_steps:
  registration_processing: true
  scan_processing: true
  session_processing: true
  neo4j_visitor_processing: true
  # ... more steps

# Neo4j configuration
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "password"

# Recommendation settings
recommendation:
  min_similarity_score: 0.3
  max_recommendations: 10
  use_langchain: false
```

## Output Files

The pipeline generates various output files:

### CSV Files
- `Registration_data_bva_25_only_valid.csv` - Current year registrations
- `df_reg_demo_this.csv` - Current year with demographics
- `session_this_filtered_valid_cols.csv` - Processed sessions

### JSON Files
- `streams.json` - Stream categories with AI descriptions
- `specializations.json` - Practice specializations
- `visitor_recommendations_[timestamp].json` - Recommendations

### Logs
- `logs/data_processing.log` - Main processing log
- `logs/processing_summary.json` - Summary statistics

## Neo4j Schema

### Node Types
- **Visitor_this_year**: Current year visitors
- **Visitor_last_year_bva/lva**: Past year visitors
- **Sessions_this_year**: Current year sessions
- **Sessions_past_year**: Past year sessions
- **Stream**: Session categories

### Relationships
- **HAS_STREAM**: Session → Stream
- **job_to_stream**: Visitor → Stream (based on job role)
- **specialization_to_stream**: Visitor → Stream (based on practice)
- **Same_Visitor**: Links same visitor across years
- **attended_session**: Past visitor → Past session

## Business Rules

The recommendation engine applies configurable business rules:

1. **Practice Type Rules**:
   - Equine/Mixed practices: Excludes small animal, feline, exotics
   - Small Animal practices: Excludes equine, farm, large animal

2. **Job Role Rules**:
   - Vets: Cannot attend nursing-specific sessions
   - Nurses: Limited to nursing, wellbeing, welfare streams

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Ensure Neo4j is running
   - Check credentials in `.env` file
   - Verify URI format (bolt://host:port)

2. **Missing Data Files**
   - Verify all input paths in config.yaml
   - Ensure JSON files are properly formatted

3. **Memory Issues**
   - Process data in batches using `--only-steps`
   - Increase Python heap size if needed

4. **API Rate Limits**
   - Enable stream description caching in config
   - Use `use_cached_descriptions: true`

## Development

### Adding New Processors
1. Create a new processor class inheriting from base patterns
2. Add to `pipeline.py`
3. Update configuration schema
4. Add command-line options in `main.py`

### Extending Neo4j Schema
1. Update node/relationship definitions in processor
2. Add constraints in `neo4j_schema.py`
3. Update documentation



## Support

For issues and questions:
- Check logs in `logs/` directory
- Review configuration in `config/config.yaml`
- Ensure all dependencies are installed
- Verify Neo4j is accessible