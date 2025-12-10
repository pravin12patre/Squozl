// Short Squeeze Finder - Frontend Application

class SqueezeFinderApp {
    constructor() {
        this.candidates = [];
        this.filteredCandidates = [];
        this.currentFilter = 'all';
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadCandidates();
        
        // Auto-refresh every 5 minutes
        setInterval(() => this.loadCandidates(), 5 * 60 * 1000);
    }
    
    bindEvents() {
        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.refreshData();
        });
        
        // Add ticker
        document.getElementById('add-ticker-btn').addEventListener('click', () => {
            this.addTicker();
        });
        
        document.getElementById('ticker-search').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.addTicker();
            }
        });
        
        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.currentFilter = e.target.dataset.filter;
                this.applyFilter();
            });
        });
        
        // Modal close
        document.getElementById('modal-close').addEventListener('click', () => {
            this.closeModal();
        });
        
        document.getElementById('detail-modal').addEventListener('click', (e) => {
            if (e.target.id === 'detail-modal') {
                this.closeModal();
            }
        });
        
        // ESC key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }
    
    async loadCandidates() {
        try {
            const response = await fetch('/api/candidates');
            const data = await response.json();
            
            this.candidates = data.candidates || [];
            this.updateStats(data);
            this.applyFilter();
            
        } catch (error) {
            console.error('Error loading candidates:', error);
            this.showError('Failed to load data. Please try again.');
        }
    }
    
    async refreshData() {
        const btn = document.getElementById('refresh-btn');
        btn.classList.add('loading');
        btn.disabled = true;
        
        try {
            await fetch('/api/refresh');
            
            // Wait a bit then reload
            setTimeout(async () => {
                await this.loadCandidates();
                btn.classList.remove('loading');
                btn.disabled = false;
            }, 3000);
            
        } catch (error) {
            console.error('Error refreshing:', error);
            btn.classList.remove('loading');
            btn.disabled = false;
        }
    }
    
    async addTicker() {
        const input = document.getElementById('ticker-search');
        const ticker = input.value.trim().toUpperCase();
        
        if (!ticker) return;
        
        const btn = document.getElementById('add-ticker-btn');
        btn.disabled = true;
        btn.textContent = '...';
        
        try {
            const response = await fetch(`/api/add/${ticker}`);
            const data = await response.json();
            
            if (data.data) {
                // Check if already exists
                const exists = this.candidates.find(c => c.ticker === ticker);
                if (!exists) {
                    this.candidates.push(data.data);
                    this.candidates.sort((a, b) => b.squeeze_score - a.squeeze_score);
                }
                this.applyFilter();
                this.showNotification(`${ticker} added successfully!`, 'success');
            } else {
                this.showNotification(data.message || 'Could not add ticker', 'error');
            }
            
            input.value = '';
            
        } catch (error) {
            console.error('Error adding ticker:', error);
            this.showNotification('Network error. Please try again.', 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Add';
        }
    }
    
    showNotification(message, type = 'info') {
        // Remove existing notification
        const existing = document.querySelector('.notification');
        if (existing) existing.remove();
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 24px;
            padding: 16px 24px;
            border-radius: 8px;
            font-weight: 500;
            z-index: 1001;
            animation: slideIn 0.3s ease;
            ${type === 'success' ? 'background: #00ff88; color: #0a0e17;' : 'background: #ff3366; color: white;'}
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    updateStats(data) {
        document.getElementById('total-candidates').textContent = data.total || 0;
        
        const highAlert = this.candidates.filter(c => c.squeeze_score >= 70).length;
        document.getElementById('high-alert').textContent = highAlert;
        
        if (data.last_updated) {
            const date = new Date(data.last_updated);
            document.getElementById('last-updated').textContent = date.toLocaleTimeString();
        }
    }
    
    applyFilter() {
        switch (this.currentFilter) {
            case 'high':
                this.filteredCandidates = this.candidates.filter(c => c.squeeze_score >= 70);
                break;
            case 'medium':
                this.filteredCandidates = this.candidates.filter(c => c.squeeze_score >= 40 && c.squeeze_score < 70);
                break;
            case 'rising':
                this.filteredCandidates = this.candidates.filter(c => c.volume_change_pct > 50);
                break;
            default:
                this.filteredCandidates = [...this.candidates];
        }
        
        this.renderCandidates();
    }
    
    renderCandidates() {
        const grid = document.getElementById('candidates-grid');
        
        if (this.filteredCandidates.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <h3>No candidates found</h3>
                    <p>Try adjusting filters or add more tickers to monitor.</p>
                </div>
            `;
            return;
        }
        
        grid.innerHTML = this.filteredCandidates.map(stock => this.createStockCard(stock)).join('');
        
        // Bind card click events
        grid.querySelectorAll('.stock-card').forEach(card => {
            card.addEventListener('click', (e) => {
                // Don't open modal if clicking remove button
                if (e.target.classList.contains('remove-btn')) return;
                
                const ticker = card.dataset.ticker;
                const stock = this.candidates.find(c => c.ticker === ticker);
                if (stock) this.openModal(stock);
            });
        });
        
        // Bind remove button events
        grid.querySelectorAll('.remove-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const ticker = btn.dataset.ticker;
                // Add visual feedback before removing
                const card = btn.closest('.stock-card');
                card.style.opacity = '0.5';
                card.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    this.removeTicker(ticker);
                }, 200);
            });
        });
    }
    
    createStockCard(stock) {
        const scoreClass = stock.squeeze_score >= 70 ? 'high' : stock.squeeze_score >= 40 ? 'medium' : '';
        const priceClass = stock.price_change_5d >= 0 ? 'positive' : 'negative';
        const volClass = stock.volume_change_pct >= 0 ? 'positive' : 'negative';
        
        return `
            <div class="stock-card ${scoreClass ? 'high-score' : ''}" data-ticker="${stock.ticker}">
                <button class="remove-btn" data-ticker="${stock.ticker}" title="Remove ${stock.ticker}">x</button>
                <div class="card-header">
                    <div class="ticker-info">
                        <span class="ticker">${stock.ticker}</span>
                        <span class="company-name">${stock.name || stock.ticker}</span>
                    </div>
                    <div class="squeeze-score ${scoreClass}">
                        ${stock.squeeze_score}
                    </div>
                </div>
                
                <div class="card-metrics">
                    <div class="metric">
                        <span class="metric-label">Price</span>
                        <span class="metric-value">$${stock.price?.toFixed(2) || 'N/A'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">5D Change</span>
                        <span class="metric-value ${priceClass}">${stock.price_change_5d >= 0 ? '+' : ''}${stock.price_change_5d?.toFixed(2) || 0}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Short % Float</span>
                        <span class="metric-value">${stock.short_percent_float?.toFixed(1) || 'N/A'}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Vol Change</span>
                        <span class="metric-value ${volClass}">${stock.volume_change_pct >= 0 ? '+' : ''}${stock.volume_change_pct?.toFixed(0) || 0}%</span>
                    </div>
                </div>
                
                <div class="card-footer">
                    <p class="thesis-preview">${stock.thesis || 'Click to view detailed analysis...'}</p>
                </div>
            </div>
        `;
    }
    
    async removeTicker(ticker) {
        try {
            const response = await fetch(`/api/remove/${ticker}`);
            const data = await response.json();
            
            // Remove from local candidates
            this.candidates = this.candidates.filter(c => c.ticker !== ticker);
            this.applyFilter();
            
            // Update stats
            document.getElementById('total-candidates').textContent = this.candidates.length;
            const highAlert = this.candidates.filter(c => c.squeeze_score >= 70).length;
            document.getElementById('high-alert').textContent = highAlert;
            
            this.showNotification(`${ticker} removed`, 'success');
            
        } catch (error) {
            console.error('Error removing ticker:', error);
            this.showNotification('Failed to remove ticker', 'error');
        }
    }
    
    openModal(stock) {
        const modal = document.getElementById('detail-modal');
        
        document.getElementById('modal-ticker').textContent = stock.ticker;
        document.getElementById('modal-name').textContent = stock.name || stock.ticker;
        
        const scoreCircle = document.getElementById('modal-score-circle');
        scoreCircle.className = 'score-circle';
        if (stock.squeeze_score >= 70) scoreCircle.classList.add('high');
        document.getElementById('modal-score').textContent = stock.squeeze_score;
        
        const priceClass = stock.price_change_5d >= 0 ? 'positive' : 'negative';
        const volClass = stock.volume_change_pct >= 0 ? 'positive' : 'negative';
        
        document.getElementById('modal-metrics').innerHTML = `
            <div class="metric">
                <span class="metric-label">Current Price</span>
                <span class="metric-value">$${stock.price?.toFixed(2) || 'N/A'}</span>
            </div>
            <div class="metric">
                <span class="metric-label">5-Day Change</span>
                <span class="metric-value ${priceClass}">${stock.price_change_5d >= 0 ? '+' : ''}${stock.price_change_5d?.toFixed(2) || 0}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Short Interest</span>
                <span class="metric-value">${stock.short_percent_float?.toFixed(1) || 'N/A'}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Days to Cover</span>
                <span class="metric-value">${stock.days_to_cover?.toFixed(1) || 'N/A'}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Volume Change</span>
                <span class="metric-value ${volClass}">${stock.volume_change_pct >= 0 ? '+' : ''}${stock.volume_change_pct?.toFixed(0) || 0}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Float Shares</span>
                <span class="metric-value">${this.formatNumber(stock.float_shares)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Market Cap</span>
                <span class="metric-value">${this.formatNumber(stock.market_cap)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Sector</span>
                <span class="metric-value">${stock.sector || 'Unknown'}</span>
            </div>
        `;
        
        document.getElementById('modal-thesis').textContent = stock.thesis || 'No thesis available.';
        
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
    
    closeModal() {
        document.getElementById('detail-modal').classList.remove('active');
        document.body.style.overflow = '';
    }
    
    formatNumber(num) {
        if (!num) return 'N/A';
        if (num >= 1e12) return (num / 1e12).toFixed(1) + 'T';
        if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
        return num.toString();
    }
    
    showError(message) {
        const grid = document.getElementById('candidates-grid');
        grid.innerHTML = `
            <div class="empty-state">
                <h3>Error</h3>
                <p>${message}</p>
            </div>
        `;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SqueezeFinderApp();
});

