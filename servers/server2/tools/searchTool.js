import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { BaseTool } from './baseTool.js';
import axios from 'axios';
import crypto from 'crypto';

const SearchSchema = z.object({
    query: z.string().describe('Search query to execute'),
    num: z
        .number()
        .int()
        .min(1)
        .max(20)
        .default(10)
        .describe('Number of results to return (1-10)'),
});

const ToolDefinition = {
    name: 'google-search',
    description: `
    🎯 Purpose:
      1. Performs web searches using Google Custom Search API with intelligent caching.
      2. Returns relevant search results with titles, links, and snippets.
      3. Automatically caches results to reduce API usage and improve response time.

    🧭 Usage Guidance:
      1. Use for finding current information on the web.
      2. Cached results are served for identical queries within 30 minutes.
      3. Combine with read-webpage tool to get full content from search results.

    📋 Prerequisites:
      1. GOOGLE_API_KEY must be set in environment variables.
      2. GOOGLE_SEARCH_ENGINE_ID must be configured.
  `,
    inputSchema: zodToJsonSchema(SearchSchema),
    annotations: {
        title: 'Google Web Search (Cached)',
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
    },
};

class SearchCache {
    constructor(ttlMinutes = 30) {
        this.cache = new Map();
        this.ttl = ttlMinutes * 60 * 1000; // Convert to milliseconds
    }

    generateKey(query, num) {
        const queryData = JSON.stringify({ query: query.toLowerCase().trim(), num });
        return crypto.createHash('md5').update(queryData).digest('hex');
    }

    get(query, num) {
        const key = this.generateKey(query, num);
        const cached = this.cache.get(key);
        
        if (cached && Date.now() - cached.timestamp < this.ttl) {
            console.log(`Cache hit for search query: ${query.substring(0, 50)}...`);
            return cached.data;
        }
        
        if (cached) {
            // Remove expired entry
            this.cache.delete(key);
            console.log(`Cache expired for search query: ${query.substring(0, 50)}...`);
        }
        
        return null;
    }

    set(query, num, data) {
        const key = this.generateKey(query, num);
        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
        console.log(`Cached search results for query: ${query.substring(0, 50)}...`);
    }

    clear() {
        const size = this.cache.size;
        this.cache.clear();
        console.log(`Cleared ${size} cached search results`);
        return size;
    }

    getStats() {
        const now = Date.now();
        let validEntries = 0;
        let expiredEntries = 0;

        for (const [key, value] of this.cache.entries()) {
            if (now - value.timestamp < this.ttl) {
                validEntries++;
            } else {
                expiredEntries++;
            }
        }

        return {
            totalEntries: this.cache.size,
            validEntries,
            expiredEntries,
            ttlMinutes: this.ttl / (60 * 1000)
        };
    }

    cleanExpired() {
        const now = Date.now();
        let cleaned = 0;

        for (const [key, value] of this.cache.entries()) {
            if (now - value.timestamp >= this.ttl) {
                this.cache.delete(key);
                cleaned++;
            }
        }

        if (cleaned > 0) {
            console.log(`Cleaned ${cleaned} expired cache entries`);
        }

        return cleaned;
    }
}

// Global cache instance
const searchCache = new SearchCache(30); // 30 minutes cache

// Clean expired entries every 10 minutes
setInterval(() => {
    searchCache.cleanExpired();
}, 10 * 60 * 1000);

export class SearchTool extends BaseTool {
    apiKey;
    searchEngineId;
    axiosInstance;

    constructor() {
        super(SearchSchema, ToolDefinition);
        
        this.apiKey = process.env.GOOGLE_API_KEY;
        this.searchEngineId = process.env.GOOGLE_SEARCH_ENGINE_ID;
        
        if (!this.apiKey) {
            throw new Error('GOOGLE_API_KEY environment variable is required');
        }
        
        if (!this.searchEngineId) {
            throw new Error('GOOGLE_SEARCH_ENGINE_ID environment variable is required');
        }

        this.axiosInstance = axios.create({
            baseURL: 'https://www.googleapis.com/customsearch/v1',
            params: {
                key: this.apiKey,
                cx: this.searchEngineId,
            },
            timeout: 10000, // 10 second timeout
        });
    }

    async process(args) {
        try {
            const { query, num } = args;
            const limitedNum = Math.min(num, 10);

            // Check cache first
            const cachedResult = searchCache.get(query, limitedNum);
            if (cachedResult) {
                return {
                    content: [
                        {
                            type: 'text',
                            text: JSON.stringify({
                                ...cachedResult,
                                cached: true,
                                cacheInfo: "Results served from cache to reduce API usage"
                            }, null, 2),
                        },
                    ],
                };
            }

            console.log(`Making Google Search API request for: ${query.substring(0, 50)}...`);

            const response = await this.axiosInstance.get('', {
                params: {
                    q: query,
                    num: limitedNum,
                },
            });

            let results;
            if (!response.data.items) {
                results = {
                    query: query,
                    results: [],
                    message: 'No search results found',
                    totalResults: '0',
                    cached: false
                };
            } else {
                const items = response.data.items.map((item) => ({
                    title: item.title,
                    link: item.link,
                    snippet: item.snippet,
                }));

                results = {
                    query: query,
                    results: items,
                    totalResults: response.data.searchInformation?.totalResults || '0',
                    cached: false
                };
            }

            // Cache the results
            searchCache.set(query, limitedNum, results);

            return {
                content: [
                    {
                        type: 'text',
                        text: JSON.stringify(results, null, 2),
                    },
                ],
            };
        } catch (error) {
            if (axios.isAxiosError(error)) {
                // Log the error for monitoring
                console.error(`Google Search API error: ${error.response?.data?.error?.message ?? error.message}`);
                
                return {
                    content: [
                        {
                            type: 'text',
                            text: `Search API error: ${
                                error.response?.data?.error?.message ?? error.message
                            }. Note: This error was not cached to allow retry.`,
                        },
                    ],
                    isError: true,
                };
            }
            
            console.error(`Search tool error: ${error.message}`);
            
            return {
                content: [
                    {
                        type: 'text',
                        text: `Error performing search: ${error instanceof Error ? error.message : String(error)}`,
                    },
                ],
                isError: true,
            };
        }
    }

    // Static methods for cache management
    static clearCache() {
        return searchCache.clear();
    }

    static getCacheStats() {
        return searchCache.getStats();
    }

    static cleanExpiredCache() {
        return searchCache.cleanExpired();
    }
}

// Export cache management functions for use in other parts of the application
export const GoogleSearchCache = {
    clear: () => SearchTool.clearCache(),
    getStats: () => SearchTool.getCacheStats(),
    cleanExpired: () => SearchTool.cleanExpiredCache()
};