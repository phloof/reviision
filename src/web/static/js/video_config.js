// Video Configuration Manager
// Loads and manages video paths from the configuration file

class VideoConfigManager {
    constructor() {
        this.config = null;
        this.defaultVideo = null;
        this.loadPromise = this.loadConfig();
    }

    async loadConfig() {
        try {
            const response = await fetch('/static/video_config.json');
            if (!response.ok) {
                throw new Error(`Failed to load video configuration: ${response.status} ${response.statusText}`);
            }
            
            this.config = await response.json();
            this.defaultVideo = this.config.default_video || 'shopping-center';
            console.log('Video configuration loaded successfully');
            return this.config;
        } catch (error) {
            console.error('Error loading video configuration:', error);
            // Set fallback configuration
            this.config = {
                default_video: 'shopping-center',
                videos: {
                    'shopping-center': '/static/videos/shopping-center.mp4',
                    'crosswalk': '/static/videos/crosswalk.mp4',
                    'store-checkout': '/static/videos/store-checkout.mp4'
                }
            };
            this.defaultVideo = 'shopping-center';
            return this.config;
        }
    }

    async getVideoPath(videoId) {
        // Make sure config is loaded
        if (!this.config) {
            await this.loadPromise;
        }
        
        if (!videoId) {
            videoId = this.defaultVideo;
        }
        
        if (this.config.videos && this.config.videos[videoId]) {
            return this.config.videos[videoId];
        }
        
        // Return default if specific video not found
        if (this.defaultVideo && this.config.videos[this.defaultVideo]) {
            console.warn(`Video '${videoId}' not found in configuration, using default`);
            return this.config.videos[this.defaultVideo];
        }
        
        // Last resort fallback
        console.error(`Video '${videoId}' not found and no default available`);
        return '/static/videos/shopping-center.mp4';
    }

    async getDefaultVideo() {
        // Make sure config is loaded
        if (!this.config) {
            await this.loadPromise;
        }
        
        return this.defaultVideo;
    }

    async getAvailableVideos() {
        // Make sure config is loaded
        if (!this.config) {
            await this.loadPromise;
        }
        
        return this.config.videos ? Object.keys(this.config.videos) : [];
    }
}

// Create singleton instance
const videoConfig = new VideoConfigManager();

// Export singleton
window.videoConfig = videoConfig; 