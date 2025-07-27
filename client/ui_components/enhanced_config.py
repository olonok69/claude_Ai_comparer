# Fixed ui_components/enhanced_config.py - Remove nested expanders

import streamlit as st
import os
import json
from datetime import datetime
import traceback
from utils.async_helpers import reset_connection_state

# Enhanced model configuration with more providers
EXTENDED_MODEL_OPTIONS = {
    'OpenAI': {
        'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
        'default_model': 'gpt-4o',
        'required_env': ['OPENAI_API_KEY'],
        'optional_env': ['OPENAI_ORG_ID'],
        'base_url': 'https://api.openai.com/v1',
        'description': 'OpenAI GPT models with advanced reasoning capabilities',
        'status_check': 'openai_status'
    },
    'Azure OpenAI': {
        'models': ['gpt-4.1','gpt-4o', 'gpt-4o-mini', 'o3-mini'],
        'default_model': 'gpt-4.1',
        'required_env': ['AZURE_API_KEY', 'AZURE_ENDPOINT', 'AZURE_DEPLOYMENT', 'AZURE_API_VERSION'],
        'optional_env': [],
        'base_url': 'Custom Azure endpoint',
        'description': 'Azure-hosted OpenAI models with enterprise features',
        'status_check': 'azure_status'
    },
    'Anthropic': {
        'models': ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307', 'claude-3-opus-20240229'],
        'default_model': 'claude-3-5-sonnet-20241022',
        'required_env': ['ANTHROPIC_API_KEY'],
        'optional_env': [],
        'base_url': 'https://api.anthropic.com',
        'description': 'Anthropic Claude models with superior reasoning and safety',
        'status_check': 'anthropic_status'
    },
    'Google Gemini': {
        'models': ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro'],
        'default_model': 'gemini-1.5-pro',
        'required_env': ['GOOGLE_API_KEY'],
        'optional_env': ['GOOGLE_PROJECT_ID'],
        'base_url': 'https://generativelanguage.googleapis.com',
        'description': 'Google Gemini models with multimodal capabilities',
        'status_check': 'google_status'
    },
    'Cohere': {
        'models': ['command-r-plus', 'command-r', 'command'],
        'default_model': 'command-r-plus',
        'required_env': ['COHERE_API_KEY'],
        'optional_env': [],
        'base_url': 'https://api.cohere.ai',
        'description': 'Cohere Command models optimized for business use cases',
        'status_check': 'cohere_status'
    },
    'Mistral AI': {
        'models': ['mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest'],
        'default_model': 'mistral-large-latest',
        'required_env': ['MISTRAL_API_KEY'],
        'optional_env': [],
        'base_url': 'https://api.mistral.ai',
        'description': 'Mistral AI models with European privacy standards',
        'status_check': 'mistral_status'
    },
    'Ollama (Local)': {
        'models': ['llama3.1:70b', 'llama3.1:8b', 'mixtral:8x7b', 'codellama:13b'],
        'default_model': 'llama3.1:8b',
        'required_env': ['OLLAMA_BASE_URL'],
        'optional_env': ['OLLAMA_API_KEY'],
        'base_url': 'http://localhost:11434',
        'description': 'Local Ollama models for privacy and cost efficiency',
        'status_check': 'ollama_status'
    }
}

def get_provider_status(provider_name, config):
    """Check if a provider is properly configured."""
    missing_vars = []
    configured_vars = []
    
    for var in config['required_env']:
        if os.getenv(var):
            configured_vars.append(var)
        else:
            missing_vars.append(var)
    
    for var in config['optional_env']:
        if os.getenv(var):
            configured_vars.append(var)
    
    status = "✅ Configured" if not missing_vars else "❌ Missing Keys"
    return status, configured_vars, missing_vars

def test_model_connection(provider_name, model_name):
    """Test connection to a specific model."""
    try:
        # Import and test based on provider
        if provider_name == "OpenAI":
            try:
                import openai
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                return True, "Connection successful"
            except ImportError:
                return False, "OpenAI library not installed. Run: pip install openai"
        
        elif provider_name == "Azure OpenAI":
            try:
                import openai
                client = openai.AzureOpenAI(
                    api_key=os.getenv("AZURE_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                    api_version=os.getenv("AZURE_API_VERSION")
                )
                response = client.chat.completions.create(
                    model=os.getenv("AZURE_DEPLOYMENT"),
                    messages=[{"role": "user", "content": "Hello"}],
                    max_completion_tokens=5
                )
                return True, "Connection successful"
            except ImportError:
                return False, "OpenAI library not installed. Run: pip install openai"
        
        elif provider_name == "Anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                response = client.messages.create(
                    model=model_name,
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                return True, "Connection successful"
            except ImportError:
                return False, "Anthropic library not installed. Run: pip install anthropic"
        
        elif provider_name == "Google Gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello")
                return True, "Connection successful"
            except ImportError:
                return False, "Google AI library not installed. Run: pip install google-generativeai"
        
        elif provider_name == "Cohere":
            try:
                import cohere
                client = cohere.Client(os.getenv("COHERE_API_KEY"))
                response = client.generate(model=model_name, prompt="Hello", max_tokens=5)
                return True, "Connection successful"
            except ImportError:
                return False, "Cohere library not installed. Run: pip install cohere"
        
        elif provider_name == "Mistral AI":
            try:
                from mistralai.client import MistralClient
                client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
                response = client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                return True, "Connection successful"
            except ImportError:
                return False, "Mistral AI library not installed. Run: pip install mistralai"
        
        elif provider_name == "Ollama (Local)":
            try:
                import requests
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                response = requests.post(
                    f"{base_url}/api/generate",
                    json={"model": model_name, "prompt": "Hello", "stream": False},
                    timeout=10
                )
                if response.status_code == 200:
                    return True, "Connection successful"
                else:
                    return False, f"Server returned status {response.status_code}"
            except requests.exceptions.ConnectionError:
                return False, "Cannot connect to Ollama server. Is it running?"
            except Exception as e:
                return False, f"Connection error: {str(e)}"
        
        else:
            return False, f"Testing not implemented for {provider_name}"
            
    except Exception as e:
        return False, str(e)

def save_model_configuration(config_data):
    """Save model configuration to file."""
    try:
        config_path = "client/model_configs.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        return True, "Configuration saved successfully"
    except Exception as e:
        return False, f"Failed to save configuration: {str(e)}"

def load_model_configuration():
    """Load model configuration from file."""
    try:
        config_path = "client/model_configs.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def create_enhanced_configuration_tab():
    """Create the enhanced Configuration tab with model management."""
    st.markdown("Configure, test, and manage multiple AI model connections.")
    
    # Load existing configurations
    saved_configs = load_model_configuration()
    
    # Main configuration modes
    config_mode = st.radio(
        "Configuration Mode",
        ["Quick Setup", "Advanced Setup", "Manage Connections"],
        horizontal=True,
        help="Choose your configuration approach"
    )
    
    if config_mode == "Quick Setup":
        create_quick_setup_section(saved_configs)
    elif config_mode == "Advanced Setup":
        create_advanced_setup_section(saved_configs)
    else:
        create_connection_management_section(saved_configs)

def create_quick_setup_section(saved_configs):
    """Create quick setup section for common providers."""
    st.subheader("🚀 Quick Setup")
    st.markdown("Fast configuration for popular AI providers.")
    
    # Provider selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_provider = st.selectbox(
            "Choose AI Provider",
            options=list(EXTENDED_MODEL_OPTIONS.keys()),
            help="Select your preferred AI provider",
            key="quick_provider_select"
        )
    
    with col2:
        provider_config = EXTENDED_MODEL_OPTIONS[selected_provider]
        status, configured_vars, missing_vars = get_provider_status(selected_provider, provider_config)
        st.metric("Status", status)
    
    # Provider information
    with st.container(border=True):
        st.markdown(f"**{selected_provider}**")
        st.markdown(provider_config['description'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Base URL:** `{provider_config['base_url']}`")
        with col2:
            st.markdown(f"**Available Models:** {len(provider_config['models'])}")
        with col3:
            st.markdown(f"**Default Model:** `{provider_config['default_model']}`")
    
    # Environment variables setup
    st.subheader("🔐 Environment Variables")
    
    with st.container(border=True):
        st.markdown("**Required Variables:**")
        
        env_changed = False
        for var in provider_config['required_env']:
            current_value = os.getenv(var, "")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                new_value = st.text_input(
                    f"{var}",
                    value=current_value,
                    type="password" if "KEY" in var else "default",
                    key=f"quick_{var}",
                    help=f"Enter your {var}"
                )
                if new_value != current_value:
                    os.environ[var] = new_value
                    env_changed = True
            
            with col2:
                if current_value:
                    st.success("✅ Set")
                else:
                    st.error("❌ Missing")
        
        # Optional variables
        if provider_config['optional_env']:
            st.markdown("**Optional Variables:**")
            for var in provider_config['optional_env']:
                current_value = os.getenv(var, "")
                new_value = st.text_input(
                    f"{var} (Optional)",
                    value=current_value,
                    type="password" if "KEY" in var else "default",
                    key=f"quick_opt_{var}",
                    help=f"Optional: {var}"
                )
                if new_value != current_value:
                    os.environ[var] = new_value
                    env_changed = True
    
    # Model selection and testing
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox(
            "Choose Model",
            options=provider_config['models'],
            index=provider_config['models'].index(provider_config['default_model']) if provider_config['default_model'] in provider_config['models'] else 0,
            help="Select the specific model to use",
            key="quick_model_select"
        )
    
    with col2:
        if st.button("🧪 Test Connection", type="primary", use_container_width=True, key="quick_test"):
            with st.spinner(f"Testing connection to {selected_provider}..."):
                success, message = test_model_connection(selected_provider, selected_model)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
    
    # Save configuration
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Configuration", use_container_width=True, key="quick_save"):
            config_data = {
                "provider": selected_provider,
                "model": selected_model,
                "configured_at": datetime.now().isoformat(),
                "environment_vars": {var: bool(os.getenv(var)) for var in provider_config['required_env'] + provider_config['optional_env']}
            }
            
            all_configs = load_model_configuration()
            all_configs[selected_provider] = config_data
            
            success, message = save_model_configuration(all_configs)
            if success:
                st.success(message)
                # Update session state
                st.session_state['params']['model_id'] = selected_provider
                st.session_state['params']['selected_model'] = selected_model
            else:
                st.error(message)
    
    with col2:
        if st.button("🔄 Set as Active", use_container_width=True, key="quick_activate"):
            st.session_state['params']['model_id'] = selected_provider
            st.session_state['params']['selected_model'] = selected_model
            reset_connection_state()
            st.success(f"Activated {selected_provider} - {selected_model}")
            st.rerun()

def create_advanced_setup_section(saved_configs):
    """Create advanced setup section for custom configurations."""
    st.subheader("🔧 Advanced Setup")
    st.markdown("Custom configuration for specialized setups.")
    
    # Custom provider setup - use checkbox instead of expander
    show_custom_provider = st.checkbox("➕ Add Custom Provider", key="show_custom_provider")
    if show_custom_provider:
        st.markdown("Configure a custom AI provider or local model.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            custom_name = st.text_input("Provider Name", placeholder="My Custom Provider", key="custom_name")
            custom_base_url = st.text_input("Base URL", placeholder="https://api.example.com/v1", key="custom_url")
            custom_api_key_name = st.text_input("API Key Environment Variable", placeholder="CUSTOM_API_KEY", key="custom_key")
        
        with col2:
            custom_model = st.text_input("Model Name", placeholder="custom-model-v1", key="custom_model")
            custom_description = st.text_area("Description", placeholder="Description of this provider...", key="custom_desc")
        
        if st.button("Add Custom Provider", key="add_custom"):
            if all([custom_name, custom_base_url, custom_api_key_name, custom_model]):
                # Add to extended options temporarily
                EXTENDED_MODEL_OPTIONS[custom_name] = {
                    'models': [custom_model],
                    'default_model': custom_model,
                    'required_env': [custom_api_key_name],
                    'optional_env': [],
                    'base_url': custom_base_url,
                    'description': custom_description,
                    'status_check': 'custom_status'
                }
                st.success(f"Added custom provider: {custom_name}")
                st.rerun()
            else:
                st.error("Please fill in all required fields")
    
    # Bulk environment variable setup - use checkbox instead of expander
    show_bulk_env = st.checkbox("🌍 Bulk Environment Setup", key="show_bulk_env")
    if show_bulk_env:
        st.markdown("Set multiple environment variables at once.")
        
        env_text = st.text_area(
            "Environment Variables",
            placeholder="""OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...""",
            height=200,
            help="Enter one variable per line in KEY=value format",
            key="bulk_env"
        )
        
        if st.button("Apply Environment Variables", key="apply_env"):
            if env_text.strip():
                applied = 0
                errors = []
                
                for line in env_text.strip().split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key and value:
                            os.environ[key] = value
                            applied += 1
                        else:
                            errors.append(f"Invalid format: {line}")
                    else:
                        errors.append(f"Missing '=' in: {line}")
                
                if applied > 0:
                    st.success(f"Applied {applied} environment variables")
                if errors:
                    st.error("Errors:\n" + "\n".join(errors))
            else:
                st.warning("Please enter environment variables")
    
    # Model parameters configuration - use checkbox instead of expander
    show_model_params = st.checkbox("⚙️ Model Parameters", key="show_model_params", value=True)
    if show_model_params:
        st.markdown("Configure default parameters for all models.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            default_max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=32000,
                value=st.session_state.get('params', {}).get('max_tokens', 4096),
                step=256,
                help="Maximum tokens in response",
                key="adv_max_tokens"
            )
        
        with col2:
            default_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.get('params', {}).get('temperature', 1.0),
                step=0.1,
                help="Creativity level (0=deterministic, 2=very creative)",
                key="adv_temperature"
            )
        
        with col3:
            default_timeout = st.number_input(
                "Timeout (seconds)",
                min_value=10,
                max_value=300,
                value=60,
                step=10,
                help="Request timeout duration",
                key="adv_timeout"
            )
        
        # Update session state
        if 'params' not in st.session_state:
            st.session_state['params'] = {}
        
        st.session_state['params'].update({
            'max_tokens': default_max_tokens,
            'temperature': default_temperature,
            'timeout': default_timeout
        })

def create_connection_management_section(saved_configs):
    """Create connection management section."""
    st.subheader("🔌 Connection Management")
    st.markdown("Manage and test existing AI model connections.")
    
    # Current active connection
    current_provider = st.session_state.get('params', {}).get('model_id')
    current_model = st.session_state.get('params', {}).get('selected_model')
    
    if current_provider:
        st.success(f"**Active Provider:** {current_provider}")
        if current_model:
            st.info(f"**Active Model:** {current_model}")
    else:
        st.warning("No active provider selected")
    
    # Connection overview
    with st.container(border=True):
        st.markdown("### 📊 Provider Status Overview")
        
        # Display as metrics grid
        providers = list(EXTENDED_MODEL_OPTIONS.keys())
        rows = [providers[i:i+3] for i in range(0, len(providers), 3)]
        
        for row in rows:
            cols = st.columns(len(row))
            for i, provider_name in enumerate(row):
                config = EXTENDED_MODEL_OPTIONS[provider_name]
                status, configured_vars, missing_vars = get_provider_status(provider_name, config)
                
                with cols[i]:
                    st.metric(
                        provider_name,
                        status,
                        f"{len(configured_vars)} vars configured",
                        delta_color="normal" if status.startswith("✅") else "inverse"
                    )
    
    # Connection testing
    st.markdown("### 🧪 Connection Testing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_provider = st.selectbox(
            "Select Provider to Test",
            options=list(EXTENDED_MODEL_OPTIONS.keys()),
            key="test_provider_select"
        )
    
    with col2:
        if test_provider:
            provider_config = EXTENDED_MODEL_OPTIONS[test_provider]
            test_model = st.selectbox(
                "Select Model",
                options=provider_config['models'],
                key="test_model_select"
            )
    
    with col3:
        if st.button("🧪 Test Connection", use_container_width=True, key="test_connection"):
            if test_provider and test_model:
                with st.spinner(f"Testing {test_provider} - {test_model}..."):
                    success, message = test_model_connection(test_provider, test_model)
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
    
    # Quick actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Set as Active Provider", use_container_width=True, key="set_active"):
            if test_provider and test_model:
                # Update session state and force reconnection
                st.session_state['params']['model_id'] = test_provider
                st.session_state['params']['selected_model'] = test_model
                reset_connection_state()
                st.success(f"Switched to {test_provider} - {test_model}")
                st.rerun()
            else:
                st.warning("Please select provider and model first")
    
    with col2:
        if st.button("💾 Save Current Config", use_container_width=True, key="save_current"):
            if test_provider and test_model:
                config_data = {
                    "provider": test_provider,
                    "model": test_model,
                    "configured_at": datetime.now().isoformat(),
                    "environment_vars": {var: bool(os.getenv(var)) for var in EXTENDED_MODEL_OPTIONS[test_provider]['required_env'] + EXTENDED_MODEL_OPTIONS[test_provider]['optional_env']}
                }
                
                all_configs = load_model_configuration()
                all_configs[test_provider] = config_data
                
                success, message = save_model_configuration(all_configs)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please select provider and model first")
    
    # Saved configurations
    if saved_configs:
        st.markdown("### 💾 Saved Configurations")
        
        for provider_name, config_data in saved_configs.items():
            # Use container with checkbox instead of expander
            show_config = st.checkbox(f"📋 Show {provider_name} Configuration", key=f"show_config_{provider_name}")
            if show_config:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Model:** {config_data.get('model', 'Unknown')}")
                    st.write(f"**Configured:** {config_data.get('configured_at', 'Unknown')[:10]}")
                
                with col2:
                    if st.button(f"Load {provider_name}", key=f"load_{provider_name}"):
                        st.session_state['params']['model_id'] = provider_name
                        st.session_state['params']['selected_model'] = config_data.get('model')
                        st.success(f"Loaded configuration for {provider_name}")
                        st.rerun()
                
                with col3:
                    if st.button(f"Delete", key=f"delete_{provider_name}"):
                        del saved_configs[provider_name]
                        save_model_configuration(saved_configs)
                        st.success(f"Deleted {provider_name} configuration")
                        st.rerun()
    
    # Environment variable manager - use checkbox instead of expander
    show_env_manager = st.checkbox("🌍 Environment Variable Manager", key="show_env_manager")
    if show_env_manager:
        st.markdown("View and manage current environment variables.")
        
        # Show relevant environment variables
        env_vars = {}
        for provider_config in EXTENDED_MODEL_OPTIONS.values():
            for var in provider_config['required_env'] + provider_config['optional_env']:
                if var not in env_vars:
                    env_vars[var] = os.getenv(var, "")
        
        # Create editable environment variables
        st.markdown("**Current Environment Variables:**")
        
        for var, value in env_vars.items():
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.text(var)
            
            with col2:
                if value:
                    masked_value = value[:8] + "..." if len(value) > 8 else "***"
                    new_value = st.text_input(
                        f"Edit {var}",
                        value=value,
                        type="password" if "KEY" in var else "default",
                        key=f"env_{var}",
                        label_visibility="collapsed"
                    )
                    if new_value != value:
                        os.environ[var] = new_value
                        st.success(f"Updated {var}")
                else:
                    new_value = st.text_input(
                        f"Set {var}",
                        placeholder="Enter value...",
                        type="password" if "KEY" in var else "default",
                        key=f"env_new_{var}",
                        label_visibility="collapsed"
                    )
                    if new_value:
                        os.environ[var] = new_value
                        st.success(f"Set {var}")
            
            with col3:
                if value:
                    st.success("✅")
                else:
                    st.error("❌")
    
    # Export/Import configurations
    st.markdown("### 📤 Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Configuration", use_container_width=True, key="export_config"):
            config_export = {
                'providers': saved_configs,
                'current_provider': current_provider,
                'current_model': current_model,
                'exported_at': datetime.now().isoformat(),
                'app_version': '2.0.0'
            }
            st.download_button(
                "💾 Download Config",
                data=json.dumps(config_export, indent=2),
                file_name=f"ai_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_config"
            )
    
    with col2:
        uploaded_config = st.file_uploader(
            "📤 Import Configuration",
            type=['json'],
            help="Upload a previously exported configuration file",
            key="import_config"
        )
        
        if uploaded_config:
            try:
                imported_config = json.load(uploaded_config)
                if 'providers' in imported_config:
                    success, message = save_model_configuration(imported_config['providers'])
                    if success:
                        st.success("Configuration imported successfully!")
                        # Auto-load the current provider if available
                        if 'current_provider' in imported_config and imported_config['current_provider']:
                            st.session_state['params']['model_id'] = imported_config['current_provider']
                            if 'current_model' in imported_config:
                                st.session_state['params']['selected_model'] = imported_config['current_model']
                        st.rerun()
                    else:
                        st.error(f"Import failed: {message}")
                else:
                    st.error("Invalid configuration file format")
            except Exception as e:
                st.error(f"Failed to import configuration: {str(e)}")
    
    # Installation helper - use checkbox instead of expander
    show_package_helper = st.checkbox("📦 Package Installation Helper", key="show_package_helper")
    if show_package_helper:
        st.markdown("Install required packages for different providers.")
        
        installation_commands = {
            'OpenAI': 'pip install openai',
            'Azure OpenAI': 'pip install openai',
            'Anthropic': 'pip install anthropic',
            'Google Gemini': 'pip install google-generativeai',
            'Cohere': 'pip install cohere',
            'Mistral AI': 'pip install mistralai',
            'Ollama (Local)': 'pip install requests (built-in)'
        }
        
        st.markdown("**Installation Commands:**")
        for provider, command in installation_commands.items():
            st.code(command, language="bash")
        
        # Bulk install command
        st.markdown("**Install All Providers:**")
        all_packages = "pip install openai anthropic google-generativeai cohere mistralai requests"
        st.code(all_packages, language="bash")
        
        if st.button("🔍 Check Installed Packages", key="check_packages"):
            installed_packages = {}
            packages_to_check = ['openai', 'anthropic', 'google.generativeai', 'cohere', 'mistralai']
            
            for package in packages_to_check:
                try:
                    __import__(package)
                    installed_packages[package] = "✅ Installed"
                except ImportError:
                    installed_packages[package] = "❌ Not installed"
            
            st.markdown("**Package Status:**")
            for package, status in installed_packages.items():
                st.write(f"{package}: {status}")

def create_provider_comparison_table():
    """Create a comparison table of all providers."""
    st.markdown("### 📊 Provider Comparison")
    
    comparison_data = []
    for provider_name, config in EXTENDED_MODEL_OPTIONS.items():
        status, configured_vars, missing_vars = get_provider_status(provider_name, config)
        comparison_data.append({
            'Provider': provider_name,
            'Models': len(config['models']),
            'Status': status,
            'Base URL': config['base_url'][:30] + "..." if len(config['base_url']) > 30 else config['base_url'],
            'Required Vars': len(config['required_env']),
            'Optional Vars': len(config['optional_env'])
        })
    
    # Display as a table
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)

# Add this to the enhanced configuration tab
def add_provider_comparison():
    """Add provider comparison section."""
    show_comparison = st.checkbox("📊 Show Provider Comparison", key="show_provider_comparison")
    if show_comparison:
        create_provider_comparison_table()