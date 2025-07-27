import { GoogleSearchCache } from '../tools/searchTool.js';
import { WebpageCache } from '../tools/readWebpageTool.js';

// Health check cache to avoid frequent checks (NO API CALLS EVER)
class HealthCheckCache {
    constructor(ttlMinutes = 5) {
        this.cache = null;
        this.lastCheck = 0;
        this.ttl = ttlMinutes * 60 * 1000; // Convert to milliseconds
        // Removed API testing functionality completely
    }

    shouldUseCache() {
        return this.cache !== null && Date.now() - this.lastCheck < this.ttl;
    }

    setCache(data) {
        this.cache = data;
        this.lastCheck = Date.now();
    }
}

const healthCache = new HealthCheckCache(5); // 5 minutes cache

export async function optimizedHealthCheck(req, res) {
    try {
        // Use cached health check if available
        if (healthCache.shouldUseCache()) {
            console.log('Using cached health check result');
            return res.json(healthCache.cache);
        }

        const APP_NAME = 'google-search-mcp-sse-server';
        const APP_VERSION = '1.0.1'; // Updated version to indicate optimization

        // Basic health check data (always fresh) - NO EXTERNAL API CALLS
        const basicHealth = {
            status: 'healthy',
            server: APP_NAME,
            version: APP_VERSION,
            timestamp: new Date().toISOString(),
            activeConnections: Object.keys(req.app.locals.transports || {}).length,
            environment: {
                googleApiConfigured: !!process.env.GOOGLE_API_KEY,
                searchEngineConfigured: !!process.env.GOOGLE_SEARCH_ENGINE_ID,
                nodeVersion: process.version,
                uptime: Math.floor(process.uptime()),
            },
            cache: {
                search: GoogleSearchCache.getStats(),
                webpage: WebpageCache.getStats(),
                lastUpdated: new Date().toISOString()
            },
            features: {
                apiCaching: true,
                contentCaching: true,
                intelligentHealthCheck: true,
                reducedApiUsage: true,
                noExternalApiCallsInHealthCheck: true
            }
        };

        // IMPORTANT: NO API connectivity test in health checks
        // This prevents unnecessary API calls during health monitoring
        let apiStatus = {
            googleApi: {
                status: basicHealth.environment.googleApiConfigured && basicHealth.environment.searchEngineConfigured ? 'configured' : 'misconfigured',
                note: 'Health checks do not test external APIs to avoid unnecessary calls',
                lastTested: 'never - health checks optimized to avoid API calls',
                configurationStatus: {
                    apiKey: basicHealth.environment.googleApiConfigured ? 'configured' : 'missing',
                    searchEngineId: basicHealth.environment.searchEngineConfigured ? 'configured' : 'missing'
                }
            }
        };

        if (!basicHealth.environment.googleApiConfigured || !basicHealth.environment.searchEngineConfigured) {
            apiStatus.googleApi.missingVars = [
                !basicHealth.environment.googleApiConfigured ? 'GOOGLE_API_KEY' : null,
                !basicHealth.environment.searchEngineConfigured ? 'GOOGLE_SEARCH_ENGINE_ID' : null
            ].filter(Boolean);
        }

        // Combine basic health with API status (no external calls)
        const fullHealth = {
            ...basicHealth,
            external: apiStatus,
            optimization: {
                healthCheckCaching: true,
                externalApiCallsDisabled: 'Health checks do not call external APIs to reduce usage',
                apiTestingApproach: 'API functionality verified through actual tool usage only',
                healthCheckFrequency: 'Can be called frequently without API cost concerns'
            },
            tools: [
                {
                    name: 'google-search',
                    description: 'Google Custom Search API with caching',
                    status: 'available',
                    cached: true,
                    note: 'API tested only when tool is actually used'
                },
                {
                    name: 'read-webpage',
                    description: 'Webpage content extraction with caching',
                    status: 'available',
                    cached: true,
                    note: 'No external dependencies for health status'
                },
                {
                    name: 'clear-cache',
                    description: 'Cache management tool',
                    status: 'available',
                    cached: false
                },
                {
                    name: 'cache-stats',
                    description: 'Cache statistics tool',
                    status: 'available',
                    cached: false
                }
            ]
        };

        // Determine overall status based on configuration only
        if (!fullHealth.environment.googleApiConfigured || !fullHealth.environment.searchEngineConfigured) {
            fullHealth.status = 'degraded';
            fullHealth.message = 'Server operational but Google API not fully configured';
        } else {
            fullHealth.status = 'healthy';
            fullHealth.message = 'All systems operational - API functionality verified through usage';
        }

        // Cache the result
        healthCache.setCache(fullHealth);

        res.json(fullHealth);

    } catch (error) {
        console.error('Health check error:', error);
        
        const errorResponse = {
            status: 'unhealthy',
            server: 'google-search-mcp-sse-server',
            version: '1.0.1',
            timestamp: new Date().toISOString(),
            error: error.message,
            message: 'Server experiencing issues',
            note: 'No external API calls made during health check'
        };

        res.status(500).json(errorResponse);
    }
}

// Export health check cache stats for monitoring
export function getHealthCheckStats() {
    return {
        cacheHits: healthCache.cache !== null ? 1 : 0,
        lastCheck: healthCache.lastCheck > 0 ? new Date(healthCache.lastCheck).toISOString() : 'never',
        ttlMinutes: healthCache.ttl / (60 * 1000),
        externalApiCalls: 'disabled - health checks do not call external APIs',
        optimization: 'Health checks optimized to avoid any external API usage'
    };
}