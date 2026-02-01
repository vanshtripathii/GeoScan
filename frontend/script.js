class ChangeDetector {
    constructor() {
        this.image1 = null;
        this.image2 = null;
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        document.getElementById('image1').addEventListener('change', (e) => {
            this.loadImage(e.target.files[0], 'preview1').then(img => this.image1 = img);
        });
        
        document.getElementById('image2').addEventListener('change', (e) => {
            this.loadImage(e.target.files[0], 'preview2').then(img => this.image2 = img);
        });
        
        document.getElementById('detectBtn').addEventListener('click', () => {
            this.detectChanges();
        });
    }
    
    async loadImage(file, canvasId) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            
            reader.onload = function(e) {
                const img = new Image();
                img.onload = function() {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    resolve(img);
                }
                img.src = e.target.result;
            }
            reader.readAsDataURL(file);
        });
    }
    
    async detectChanges() {
        if (!this.image1 || !this.image2) {
            alert('Please upload both images');
            return;
        }
        
        try {
            // Convert images to base64
            const image1Base64 = this.canvasToBase64('preview1');
            const image2Base64 = this.canvasToBase64('preview2');
            
            const response = await fetch('http://localhost:5000/detect-changes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image1: image1Base64,
                    image2: image2Base64
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result);
            } else {
                alert('Error: ' + result.error);
            }
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error detecting changes');
        }
    }
    
    canvasToBase64(canvasId) {
        const canvas = document.getElementById(canvasId);
        return canvas.toDataURL('image/png');
    }
    
    displayResults(result) {
        // Display change percentage
        const percentageElement = document.getElementById('changePercentage');
        percentageElement.textContent = `Change detected: ${(result.change_percentage * 100).toFixed(2)}%`;
        
        // Display change mask
        const resultCanvas = document.getElementById('resultCanvas');
        const ctx = resultCanvas.getContext('2d');
        
        const img = new Image();
        img.onload = function() {
            resultCanvas.width = img.width;
            resultCanvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        }
        img.src = 'data:image/png;base64,' + result.change_mask;
    }
}

// Initialize the application
new ChangeDetector();