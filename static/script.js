// Additional JavaScript functionality for RFID Access Control System

// Global variables
let autoRefreshInterval;
let isAutoRefreshEnabled = true;

// Initialize additional functionality
document.addEventListener('DOMContentLoaded', function() {
    initializeAutoRefresh();
    initializeTooltips();
    initializeKeyboardShortcuts();
});

// Auto-refresh functionality
function initializeAutoRefresh() {
    // Toggle auto-refresh with double-click on refresh button
    const refreshBtn = document.querySelector('.refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('dblclick', toggleAutoRefresh);
    }
}

function toggleAutoRefresh() {
    isAutoRefreshEnabled = !isAutoRefreshEnabled;
    const indicator = document.getElementById('status-indicator');
    
    if (isAutoRefreshEnabled) {
        indicator.classList.add('status-pulse');
        showNotification('Auto-refresh enabled', 'success');
    } else {
        indicator.classList.remove('status-pulse');
        showNotification('Auto-refresh disabled', 'warning');
    }
}

// Tooltip initialization
function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Keyboard shortcuts
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl+R or F5: Refresh data
        if ((e.ctrlKey && e.key === 'r') || e.key === 'F5') {
            e.preventDefault();
            refreshData();
        }
        
        // Ctrl+A: Toggle auto-refresh
        if (e.ctrlKey && e.key === 'a') {
            e.preventDefault();
            toggleAutoRefresh();
        }
        
        // Escape: Close any open modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) bsModal.hide();
            });
        }
    });
}

// Notification system
function showNotification(message, type = 'info', duration = 3000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
}

// Enhanced image modal functionality
function showImageModal(filename, cardNumber, timestamp, uploaded, s3Location) {
    // Update modal content
    document.getElementById('modalImage').src = `/image/${filename}`;
    document.getElementById('modalFileName').textContent = filename;
    document.getElementById('modalCardNumber').textContent = cardNumber;
    document.getElementById('modalTimestamp').textContent = new Date(timestamp * 1000).toLocaleString();
    
    const uploadStatus = getUploadStatus(uploaded);
    document.getElementById('modalUploadStatus').innerHTML = 
        `<span class="badge bg-${uploadStatus.color}">${uploadStatus.text}</span>`;
    
    document.getElementById('modalS3Location').textContent = s3Location || 'Not uploaded';
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('imageModal'));
    modal.show();
    
    // Add download functionality
    const modalImage = document.getElementById('modalImage');
    modalImage.onclick = function() {
        downloadImage(filename);
    };
    modalImage.style.cursor = 'pointer';
    modalImage.title = 'Click to download';
}

// Download image functionality
function downloadImage(filename) {
    const link = document.createElement('a');
    link.href = `/image/${filename}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showNotification(`Downloaded ${filename}`, 'success');
}

// Enhanced error handling
function handleApiError(error, context) {
    console.error(`Error in ${context}:`, error);
    showNotification(`Error in ${context}: ${error.message}`, 'danger');
}

// Connection status monitoring
function updateConnectionStatus(online = true) {
    const indicator = document.getElementById('status-indicator');
    const status = document.getElementById('connection-status');
    
    if (online) {
        indicator.className = 'fas fa-circle text-success';
        status.textContent = 'Online';
        indicator.classList.remove('status-pulse');
    } else {
        indicator.className = 'fas fa-circle text-danger status-pulse';
        status.textContent = 'Offline';
    }
}

// Enhanced refresh with loading states
async function refreshData() {
    if (isRefreshing) return;
    isRefreshing = true;

    const refreshBtn = document.querySelector('.refresh-btn i');
    const originalClass = refreshBtn.className;
    refreshBtn.className = 'fas fa-spinner fa-spin';

    try {
        // Update connection status
        updateConnectionStatus();
        
        // Fetch and display scans
        await fetchAndDisplayScans();
        
        // Fetch and display images
        await fetchAndDisplayImages();
        
        showNotification('Data refreshed successfully', 'success', 2000);
        
    } catch (error) {
        console.error('Error refreshing data:', error);
        updateConnectionStatus(false);
        showNotification('Failed to refresh data', 'danger');
    } finally {
        isRefreshing = false;
        refreshBtn.className = originalClass;
    }
}

// Export functions for global access
window.showNotification = showNotification;
window.downloadImage = downloadImage;
window.toggleAutoRefresh = toggleAutoRefresh;
