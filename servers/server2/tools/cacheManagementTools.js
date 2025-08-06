import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { BaseTool } from './baseTool.js';
import { GoogleSearchCache } from './searchTool.js';
import { WebpageCache } from './readWebpageTool.js';

// Schema for cache management tools
const ClearCacheSchema = z.object({
    cacheType: z.enum(['search', 'webpage', 'all']).default('all').describe('Type of cache to clear: search, webpage, or all'),
});

const CacheStatsSchema = z.object({
    detailed: z.boolean().default(false).describe('Include detailed cache statistics'),
});

// Tool definitions
const ClearCacheToolDefinition = {
    name: 'clear-cache',
    description: `
    🎯 Purpose:
      1. Clears cached API responses and webpage content.
      2. Use when you need fresh data from external APIs.
      3. Helps manage memory usage by clearing old cache entries.

    🧭 Usage Guidance:
      1. Use 'search' to clear only Google Search API cache.
      2. Use 'webpage' to clear only webpage content cache.
      3. Use 'all' to clear both caches (default).
      4. Cache clearing is immediate and cannot be undone.

    📦 Cache Types:
      - search: Google Search API responses (30 min TTL)
      - webpage: Webpage content cache (2 hour TTL)
      - all: Both search and webpage caches
  `,
    inputSchema: zodToJsonSchema(ClearCacheSchema),
    annotations: {
        title: 'Clear API/Content Cache',
        readOnlyHint: false,
        destructiveHint: true,
        idempotentHint: true,
        openWorldHint: false,
    },
};

const CacheStatsToolDefinition = {
    name: 'cache-stats',
    description: `
    🎯 Purpose:
      1. Provides statistics about current cache usage.
      2. Shows cache hit rates and memory efficiency.
      3. Helps monitor API usage reduction through caching.

    🧭 Usage Guidance:
      1. Use to monitor cache performance and effectiveness.
      2. Detailed stats show cache sizes and TTL information.
      3. Helps decide when cache clearing might be beneficial.

    📊 Statistics Provided:
      - Total cached entries (search + webpage)
      - Valid vs expired entries
      - Cache hit effectiveness
      - Memory usage estimates
  `,
    inputSchema: zodToJsonSchema(CacheStatsSchema),
    annotations: {
        title: 'Get Cache Statistics',
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
    },
};

export class ClearCacheTool extends BaseTool {
    constructor() {
        super(ClearCacheSchema, ClearCacheToolDefinition);
    }

    async process(args) {
        try {
            const { cacheType } = args;
            let clearedEntries = 0;
            const results = {
                timestamp: new Date().toISOString(),
                cacheType: cacheType,
                cleared: {}
            };

            switch (cacheType) {
                case 'search':
                    results.cleared.search = GoogleSearchCache.clear();
                    clearedEntries = results.cleared.search;
                    break;
                
                case 'webpage':
                    results.cleared.webpage = WebpageCache.clear();
                    clearedEntries = results.cleared.webpage;
                    break;
                
                case 'all':
                default:
                    results.cleared.search = GoogleSearchCache.clear();
                    results.cleared.webpage = WebpageCache.clear();
                    clearedEntries = results.cleared.search + results.cleared.webpage;
                    break;
            }

            results.totalCleared = clearedEntries;
            results.message = `Successfully cleared ${clearedEntries} cache entries`;

            if (clearedEntries === 0) {
                results.message = "No cache entries to clear - caches were already empty";
            }

            return {
                content: [
                    {
                        type: 'text',
                        text: JSON.stringify(results, null, 2),
                    },
                ],
            };
        } catch (error) {
            return {
                content: [
                    {
                        type: 'text',
                        text: `Error clearing cache: ${error instanceof Error ? error.message : String(error)}`,
                    },
                ],
                isError: true,
            };
        }
    }
}

export class CacheStatsTool extends BaseTool {
    constructor() {
        super(CacheStatsSchema, CacheStatsToolDefinition);
    }

    async process(args) {
        try {
            const { detailed } = args;
            
            const searchStats = GoogleSearchCache.getStats();
            const webpageStats = WebpageCache.getStats();
            
            const results = {
                timestamp: new Date().toISOString(),
                overview: {
                    totalCachedItems: searchStats.totalEntries + webpageStats.totalEntries,
                    totalValidItems: searchStats.validEntries + webpageStats.validEntries,
                    totalExpiredItems: searchStats.expiredEntries + webpageStats.expiredEntries,
                    estimatedMemoryKB: (webpageStats.estimatedSizeKB || 0) + (searchStats.totalEntries * 2), // Rough estimate
                },
                caches: {
                    search: {
                        type: "Google Search API Cache",
                        ...searchStats,
                        description: "Caches Google Search API responses to reduce API calls"
                    },
                    webpage: {
                        type: "Webpage Content Cache",
                        ...webpageStats,
                        description: "Caches extracted webpage content to reduce server load"
                    }
                }
            };

            // Add efficiency metrics
            const totalItems = results.overview.totalCachedItems;
            if (totalItems > 0) {
                results.efficiency = {
                    cacheUtilization: Math.round((results.overview.totalValidItems / totalItems) * 100) + "%",
                    apiCallsAvoided: results.overview.totalValidItems,
                    memoryEfficiency: totalItems > 0 ? Math.round(results.overview.estimatedMemoryKB / totalItems) + " KB per item" : "N/A"
                };
            }

            // Add recommendations
            results.recommendations = [];
            
            if (results.overview.totalExpiredItems > 10) {
                results.recommendations.push("Consider clearing expired cache entries to free memory");
            }
            
            if (results.overview.totalValidItems > 100) {
                results.recommendations.push("Cache is working effectively - good API usage reduction");
            }
            
            if (results.overview.estimatedMemoryKB > 5000) {
                results.recommendations.push("High memory usage detected - consider clearing webpage cache");
            }

            // Remove detailed stats if not requested
            if (!detailed) {
                delete results.caches.search.description;
                delete results.caches.webpage.description;
            } else {
                // Add more detailed information
                results.detailed = {
                    cacheHitRate: {
                        description: "Estimated percentage of requests served from cache",
                        estimated: totalItems > 0 ? Math.round((results.overview.totalValidItems / (totalItems + results.overview.totalValidItems)) * 100) + "%" : "0%"
                    },
                    nextCleanup: "Expired entries are automatically cleaned periodically",
                    cacheLifecycle: {
                        search: `Search results cached for ${searchStats.ttlMinutes} minutes`,
                        webpage: `Webpage content cached for ${webpageStats.ttlHours} hours`
                    }
                };
            }

            return {
                content: [
                    {
                        type: 'text',
                        text: JSON.stringify(results, null, 2),
                    },
                ],
            };
        } catch (error) {
            return {
                content: [
                    {
                        type: 'text',
                        text: `Error getting cache statistics: ${error instanceof Error ? error.message : String(error)}`,
                    },
                ],
                isError: true,
            };
        }
    }
}