import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { BaseTool } from './baseTool.js';
import axios from 'axios';
import * as cheerio from 'cheerio';
import crypto from 'crypto';

const ReadWebpageSchema = z.object({
    url: z.string().url().describe('URL of the webpage to read and extract content from'),
    skipCache: z.boolean().optional().default(false).describe('Skip cache and fetch fresh content'),
});

const ToolDefinition = {
    name: 'read-webpage',
    description: `
    🎯 Purpose:
      1. Fetches and extracts text content from web pages with intelligent caching.
      2. Removes scripts, styles, and other non-content elements.
      3. Returns clean, readable text content with title and URL.
      4. Automatically caches page content to reduce server load and improve speed.

    🧭 Usage Guidance:
      1. Use after google-search to get full content from interesting results.
      2. Cached content is served for identical URLs within 2 hours.
      3. Use skipCache=true only when you need the absolute latest content.
      4. Ideal for reading articles, blog posts, and documentation.

    📦 Returns:
      1. Page title, clean text content, and source URL.
      2. Content is processed to remove formatting and focus on readable text.
      3. Cache information indicates if content was served from cache.
  `,
    inputSchema: zodToJsonSchema(ReadWebpageSchema),
    annotations: {
        title: 'Read Webpage Content (Cached)',
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
    },
};

class WebpageCacheClass {
    constructor(ttlHours = 2) {
        this.cache = new Map();
        this.ttl = ttlHours * 60 * 60 * 1000; // Convert to milliseconds
        this.maxCacheSize = 1000; // Maximum number of cached pages
    }

    generateKey(url) {
        // Normalize URL by removing common tracking parameters
        const cleanUrl = this.cleanUrl(url);
        return crypto.createHash('md5').update(cleanUrl).digest('hex');
    }

    cleanUrl(url) {
        try {
            const urlObj = new URL(url);
            
            // Remove common tracking parameters
            const trackingParams = [
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                'fbclid', 'gclid', 'msclkid', '_ga', '_gl', 'ref', 'source'
            ];
            
            trackingParams.forEach(param => {
                urlObj.searchParams.delete(param);
            });
            
            return urlObj.toString();
        } catch (e) {
            // If URL parsing fails, use original URL
            return url;
        }
    }

    get(url) {
        const key = this.generateKey(url);
        const cached = this.cache.get(key);
        
        if (cached && Date.now() - cached.timestamp < this.ttl) {
            console.log(`Cache hit for webpage: ${url.substring(0, 80)}...`);
            // Update access time for LRU behavior
            cached.lastAccessed = Date.now();
            return cached.data;
        }
        
        if (cached) {
            // Remove expired entry
            this.cache.delete(key);
            console.log(`Cache expired for webpage: ${url.substring(0, 80)}...`);
        }
        
        return null;
    }

    set(url, data) {
        // Check cache size and remove oldest entries if necessary
        if (this.cache.size >= this.maxCacheSize) {
            this.evictOldest();
        }

        const key = this.generateKey(url);
        this.cache.set(key, {
            data,
            timestamp: Date.now(),
            lastAccessed: Date.now(),
            url: url
        });
        console.log(`Cached webpage content: ${url.substring(0, 80)}...`);
    }

    evictOldest() {
        let oldestKey = null;
        let oldestTime = Date.now();

        for (const [key, value] of this.cache.entries()) {
            if (value.lastAccessed < oldestTime) {
                oldestTime = value.lastAccessed;
                oldestKey = key;
            }
        }

        if (oldestKey) {
            this.cache.delete(oldestKey);
            console.log('Evicted oldest cached webpage due to size limit');
        }
    }

    clear() {
        const size = this.cache.size;
        this.cache.clear();
        console.log(`Cleared ${size} cached webpages`);
        return size;
    }

    getStats() {
        const now = Date.now();
        let validEntries = 0;
        let expiredEntries = 0;
        let totalSize = 0;

        for (const [key, value] of this.cache.entries()) {
            if (now - value.timestamp < this.ttl) {
                validEntries++;
            } else {
                expiredEntries++;
            }
            
            // Estimate size (rough calculation)
            if (value.data && value.data.text) {
                totalSize += value.data.text.length;
            }
        }

        return {
            totalEntries: this.cache.size,
            validEntries,
            expiredEntries,
            ttlHours: this.ttl / (60 * 60 * 1000),
            maxCacheSize: this.maxCacheSize,
            estimatedSizeKB: Math.round(totalSize / 1024)
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
            console.log(`Cleaned ${cleaned} expired webpage cache entries`);
        }

        return cleaned;
    }
}

// Global cache instance
const webpageCache = new WebpageCacheClass(2); // 2 hours cache

// Clean expired entries every 30 minutes
setInterval(() => {
    webpageCache.cleanExpired();
}, 30 * 60 * 1000);

export class ReadWebpageTool extends BaseTool {
    constructor() {
        super(ReadWebpageSchema, ToolDefinition);
    }

    async process(args) {
        try {
            const { url, skipCache = false } = args;

            // Check cache first (unless skipCache is true)
            if (!skipCache) {
                const cachedResult = webpageCache.get(url);
                if (cachedResult) {
                    return {
                        content: [
                            {
                                type: 'text',
                                text: JSON.stringify({
                                    ...cachedResult,
                                    cached: true,
                                    cacheInfo: "Content served from cache to reduce server load"
                                }, null, 2),
                            },
                        ],
                    };
                }
            }

            console.log(`Fetching webpage content: ${url.substring(0, 80)}...`);

            const response = await axios.get(url, {
                timeout: 15000, // 15 second timeout
                headers: {
                    'User-Agent': 'Mozilla/5.0 (compatible; Google Search MCP Server/1.0)',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                },
                maxRedirects: 5,
                validateStatus: function (status) {
                    return status >= 200 && status < 400; // Accept 2xx and 3xx status codes
                }
            });

            const $ = cheerio.load(response.data);

            // Remove script and style elements and other non-content
            $('script, style, nav, header, footer, aside, .advertisement, .ads, .sidebar, .menu, .navigation, .social-share, .comments').remove();

            // Extract metadata
            const title = $('title').text().trim() || $('h1').first().text().trim() || 'No title found';
            const description = $('meta[name="description"]').attr('content') || '';
            const author = $('meta[name="author"]').attr('content') || $('[rel="author"]').text().trim() || '';

            // Extract main content
            let mainContent = '';
            
            // Try to find main content areas
            const contentSelectors = [
                'main', 'article', '.content', '.main-content', '.post-content', 
                '.entry-content', '.article-content', '#content', '.story-body'
            ];
            
            let contentFound = false;
            for (const selector of contentSelectors) {
                const element = $(selector);
                if (element.length > 0 && element.text().trim().length > 100) {
                    mainContent = element.text().trim();
                    contentFound = true;
                    break;
                }
            }
            
            // Fallback to body if no main content area found
            if (!contentFound) {
                mainContent = $('body').text().trim();
            }
            
            // Clean up whitespace
            mainContent = mainContent.replace(/\s+/g, ' ').trim();

            const content = {
                title,
                description,
                author,
                text: mainContent,
                url: url,
                contentLength: mainContent.length,
                fetchedAt: new Date().toISOString(),
                cached: false
            };

            // Limit content length to prevent token overflow
            if (content.text.length > 15000) {
                content.text = content.text.substring(0, 15000) + '... [Content truncated to prevent token overflow]';
                content.truncated = true;
                content.originalLength = mainContent.length;
            }

            // Cache the results (unless skipCache was requested)
            if (!skipCache) {
                webpageCache.set(url, content);
            }

            return {
                content: [
                    {
                        type: 'text',
                        text: JSON.stringify(content, null, 2),
                    },
                ],
            };
        } catch (error) {
            if (axios.isAxiosError(error)) {
                console.error(`Webpage fetch error for ${args.url}: ${error.message}`);
                
                let errorMessage = `Webpage fetch error: ${error.message}`;
                
                if (error.response) {
                    errorMessage += `. Status: ${error.response.status}`;
                    if (error.response.status === 403) {
                        errorMessage += ' (Access forbidden - site may block automated requests)';
                    } else if (error.response.status === 404) {
                        errorMessage += ' (Page not found)';
                    } else if (error.response.status >= 500) {
                        errorMessage += ' (Server error)';
                    }
                } else if (error.code === 'ECONNABORTED') {
                    errorMessage += ' (Request timeout - site may be slow)';
                } else if (error.code === 'ENOTFOUND') {
                    errorMessage += ' (Domain not found)';
                }
                
                return {
                    content: [
                        {
                            type: 'text',
                            text: errorMessage + '. Note: This error was not cached.',
                        },
                    ],
                    isError: true,
                };
            }
            
            console.error(`Read webpage tool error: ${error.message}`);
            
            return {
                content: [
                    {
                        type: 'text',
                        text: `Error reading webpage: ${error instanceof Error ? error.message : String(error)}`,
                    },
                ],
                isError: true,
            };
        }
    }

    // Static methods for cache management
    static clearCache() {
        return webpageCache.clear();
    }

    static getCacheStats() {
        return webpageCache.getStats();
    }

    static cleanExpiredCache() {
        return webpageCache.cleanExpired();
    }
}

// Export cache management functions for use in other parts of the application
export const WebpageCache = {
    clear: () => ReadWebpageTool.clearCache(),
    getStats: () => ReadWebpageTool.getCacheStats(),
    cleanExpired: () => ReadWebpageTool.cleanExpiredCache()
};