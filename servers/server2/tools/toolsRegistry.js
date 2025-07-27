// Import the registration function
import { registerTool } from './index.js';

// Import tool classes
import { SearchTool } from './searchTool.js';
import { ReadWebpageTool } from './readWebpageTool.js';
import { ClearCacheTool, CacheStatsTool } from './cacheManagementTools.js';

// Register all tools with caching capabilities
console.log('Registering Google Search MCP tools with caching optimization...');

// Core search and content tools (with caching)
registerTool(new SearchTool());
registerTool(new ReadWebpageTool());

// Cache management tools
registerTool(new ClearCacheTool());
registerTool(new CacheStatsTool());

console.log('✅ All tools registered successfully:');
console.log('  - google-search (with 30min cache)');
console.log('  - read-webpage (with 2hr cache)');
console.log('  - clear-cache (cache management)');
console.log('  - cache-stats (cache monitoring)');
console.log('🚀 API usage optimization active!');