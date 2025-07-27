#!/usr/bin/env node
import dotenv from 'dotenv';
dotenv.config();

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { 
    CallToolRequestSchema, 
    GetPromptRequestSchema, 
    ListPromptsRequestSchema, 
    ListToolsRequestSchema 
} from '@modelcontextprotocol/sdk/types.js';
import express from 'express';

// Import Google Search tools and prompts
import { getTools, handleToolCall } from './tools/index.js';
import { getPrompts, getPromptMessages } from './prompts/index.js';

// Import tool and prompt registries to auto-register everything
import './tools/toolsRegistry.js';
import './prompts/promptsRegistry.js';

// Import optimized health check
import { optimizedHealthCheck, getHealthCheckStats } from './utils/optimizedHealthCheck.js';

const APP_NAME = 'google-search-mcp-sse-server';
const APP_VERSION = '1.0.1'; // Updated version indicating optimization
const PORT = parseInt(process.env.PORT || '8002');
const HOST = process.env.HOST || '0.0.0.0';

// Create MCP server
const server = new Server({
    name: APP_NAME,
    version: APP_VERSION,
}, {
    capabilities: {
        tools: {},
        prompts: {},
        resources: {},
    },
});

// Handler for listing tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
        tools: getTools(),
    };
});

// Handler for calling tools
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    
    // Log tool usage for monitoring (but don't log sensitive data)
    const toolName = name;
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] Tool called: ${toolName}`);
    
    try {
        const result = await handleToolCall(name, args);
        
        // Log if result came from cache (for monitoring)
        if (result.content && result.content[0] && result.content[0].text) {
            try {
                const parsedContent = JSON.parse(result.content[0].text);
                if (parsedContent.cached === true) {
                    console.log(`[${timestamp}] Tool ${toolName}: served from cache ✓`);
                } else if (parsedContent.cached === false) {
                    console.log(`[${timestamp}] Tool ${toolName}: fresh API call made`);
                }
            } catch (e) {
                // Not JSON content, that's fine
            }
        }
        
        return result;
    } catch (error) {
        console.error(`[${timestamp}] Tool ${toolName} error:`, error.message);
        throw error;
    }
});

// Handler for listing prompts
server.setRequestHandler(ListPromptsRequestSchema, async () => {
    return {
        prompts: getPrompts(),
    };
});

// Handler for getting specific prompt
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    return getPromptMessages(name, args);
});

// To support multiple simultaneous connections we have a lookup object from
// sessionId to transport
const transports = {};

const app = express();
app.use(express.json());

// Store transports reference in app for health check
app.locals.transports = transports;

// CORS middleware
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    
    if (req.method === 'OPTIONS') {
        res.sendStatus(200);
        return;
    }
    next();
});

// Request logging middleware (minimal)
app.use((req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
        const duration = Date.now() - start;
        if (req.path !== '/health' || duration > 1000) { // Only log health checks if they're slow
            console.log(`${req.method} ${req.path} - ${res.statusCode} (${duration}ms)`);
        }
    });
    next();
});

const router = express.Router();

// Endpoint for the client to use for sending messages
const POST_ENDPOINT = "/messages";

router.post(POST_ENDPOINT, async (req, res) => {
    const sessionId = req.query.sessionId;
    if (typeof (sessionId) != "string") {
        res.status(400).send({ message: "Bad session id." });
        return;
    }
    
    const transport = transports[sessionId];
    if (!transport) {
        res.status(400).send({ message: "No transport found for sessionId." });
        return;
    }
    
    // IMPORTANT! Use the three-parameter version to avoid stream errors
    await transport.handlePostMessage(req, res, req.body);
    return;
});

// SSE endpoint - this is what the client connects to first
router.get("/sse", async (req, res) => {
    try {
        console.log('New SSE connection established');
        
        // Create a new transport to connect and send an endpoint event
        // containing a URI for the client to use for sending messages
        const transport = new SSEServerTransport(POST_ENDPOINT, res);
        
        // Store the transport for message routing
        transports[transport.sessionId] = transport;
        
        // Clean up when connection closes
        res.on("close", () => {
            delete transports[transport.sessionId];
            console.log(`SSE connection closed: ${transport.sessionId}`);
        });
        
        res.on("error", (err) => {
            console.error("SSE connection error:", err);
            delete transports[transport.sessionId];
        });
        
        // Connect the MCP server to this transport
        await server.connect(transport);
        
        return;
    } catch (error) {
        console.error("Error setting up SSE connection:", error);
        res.status(500).send("Internal Server Error");
    }
});

// Optimized health check endpoint
router.get("/health", optimizedHealthCheck);

// Additional monitoring endpoints
router.get("/health/detailed", async (req, res) => {
    try {
        const healthStats = getHealthCheckStats();
        const { GoogleSearchCache } = await import('./tools/searchTool.js');
        const { WebpageCache } = await import('./tools/readWebpageTool.js');
        
        const detailedStats = {
            timestamp: new Date().toISOString(),
            server: {
                name: APP_NAME,
                version: APP_VERSION,
                uptime: Math.floor(process.uptime()),
                memory: process.memoryUsage(),
                activeConnections: Object.keys(transports).length
            },
            healthCheck: healthStats,
            caching: {
                search: GoogleSearchCache.getStats(),
                webpage: WebpageCache.getStats()
            },
            optimization: {
                features: [
                    'API Response Caching',
                    'Webpage Content Caching', 
                    'Intelligent Health Checks',
                    'Automatic Cache Cleanup',
                    'Request Deduplication'
                ],
                benefits: [
                    'Reduced API costs',
                    'Faster response times',
                    'Lower server load',
                    'Better reliability'
                ]
            }
        };
        
        res.json(detailedStats);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Cache management endpoint (for debugging)
router.get("/cache/clear", async (req, res) => {
    try {
        const { GoogleSearchCache } = await import('./tools/searchTool.js');
        const { WebpageCache } = await import('./tools/readWebpageTool.js');
        
        const searchCleared = GoogleSearchCache.clear();
        const webpageCleared = WebpageCache.clear();
        
        res.json({
            message: 'Caches cleared successfully',
            cleared: {
                search: searchCleared,
                webpage: webpageCleared,
                total: searchCleared + webpageCleared
            },
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.use('/', router);

// Graceful shutdown handler
async function gracefulShutdown(signal) {
    console.log(`\n${signal} received. Starting graceful shutdown...`);
    
    try {
        // Close all active transports
        const transportIds = Object.keys(transports);
        console.log(`Closing ${transportIds.length} active connections...`);
        
        for (const [sessionId, transport] of Object.entries(transports)) {
            try {
                await transport.close();
            } catch (error) {
                console.error(`Error closing transport ${sessionId}:`, error);
            }
        }
        
        // Close the server
        await server.close();
        console.log('Server closed successfully');
        
        process.exit(0);
    } catch (error) {
        console.error('Error during shutdown:', error);
        process.exit(1);
    }
}

// Start the server
async function main() {
    try {
        console.log(`🚀 Starting ${APP_NAME} v${APP_VERSION}...`);
        console.log('📈 Optimization features enabled:');
        console.log('  • API response caching (30min for search, 2hr for content)');
        console.log('  • Intelligent health checks (5min cache, 30min API tests)');
        console.log('  • Automatic cache cleanup');
        console.log('  • Request deduplication');
        
        // Verify Google API credentials
        if (!process.env.GOOGLE_API_KEY) {
            console.warn('⚠️  Warning: GOOGLE_API_KEY not found in environment variables');
            console.log('   Server will run with caching but API calls will fail');
        }
        
        if (!process.env.GOOGLE_SEARCH_ENGINE_ID) {
            console.warn('⚠️  Warning: GOOGLE_SEARCH_ENGINE_ID not found in environment variables');
            console.log('   Server will run with caching but API calls will fail');
        }
        
        if (process.env.GOOGLE_API_KEY && process.env.GOOGLE_SEARCH_ENGINE_ID) {
            console.log('✅ Google API credentials configured');
        }
        
        app.listen(PORT, HOST, () => {
            console.log(`\n🌐 Google Search MCP Server running on http://${HOST}:${PORT}`);
            console.log(`📊 Health check: http://${HOST}:${PORT}/health`);
            console.log(`🔍 Detailed stats: http://${HOST}:${PORT}/health/detailed`);
            console.log(`🧹 Clear cache: http://${HOST}:${PORT}/cache/clear`);
            console.log('\n📋 Available MCP Tools:');
            console.log('  • google-search (cached for 30 minutes)');
            console.log('  • read-webpage (cached for 2 hours)'); 
            console.log('  • clear-cache (cache management)');
            console.log('  • cache-stats (monitoring)');
            console.log('\n💡 Benefits: Reduced API usage, faster responses, lower costs');
        });
        
    } catch (error) {
        console.error('❌ Error starting server:', error);
        process.exit(1);
    }
}

// Handle graceful shutdown
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    gracefulShutdown('UNCAUGHT_EXCEPTION');
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    gracefulShutdown('UNHANDLED_REJECTION');
});

// Start the server
main();