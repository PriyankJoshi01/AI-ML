import os
import torch
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from emotion_agent import analyze_call
import io
from PIL import Image  # Already available in Gradio environments

def fig_to_image(fig):
    """Convert Matplotlib figure to a PIL image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

# Use CPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
device = torch.device('cpu')
print(f"Device set to use {device}")

def run_live_analysis(audio_file):
    print("\n🎧 Starting analysis...")
    result = analyze_call(audio_file)

    if result is None or len(result) != 3:
        return "❌ Analysis failed - Please try again with a valid audio file", None, None, "No transcription available."

    emotion_history, final_percentages, full_transcript = result

    # Timing Analysis
    total_segments = len(emotion_history)
    segment_duration = 30  # seconds
    total_duration = total_segments * segment_duration
    timing_info = f"📊 Analysis completed in {total_segments} segments over {total_duration//60}m {total_duration%60}s"

    # Enhanced Bar chart for emotion percentages
    plt.style.use('default')
    bar_fig, bar_ax = plt.subplots(figsize=(12, 6))
    bar_fig.patch.set_facecolor('#ffffff')
    
    emotions = list(final_percentages.keys())
    values = list(final_percentages.values())
    
    # Professional color palette - ensure enough colors
    colors = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444', '#06b6d4', '#84cc16']
    # Extend colors if we have more emotions than colors  
    while len(colors) < len(emotions):
        colors.extend(colors)

    bar_colors = [colors[i % len(colors)] for i in range(len(emotions))]
    
    bars = bar_ax.barh(emotions, values, color=bar_colors, height=0.6)
    
    # Add value labels with better formatting
    for i, bar in enumerate(bars):
        width = bar.get_width()
        bar_ax.text(
            width + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%", 
            va='center', 
            ha='left',
            fontsize=11,
            fontweight='600',
            color='#374151'
        )
    
    bar_ax.set_xlim([0, max(values) + 8])
    bar_ax.set_title("Emotion Analysis Results", fontsize=16, fontweight='bold', color='#1f2937', pad=20)
    bar_ax.set_xlabel("Percentage (%)", fontsize=12, color='#6b7280')
    bar_ax.invert_yaxis()
    
    # Enhanced styling
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    bar_ax.spines['left'].set_color('#e5e7eb')
    bar_ax.spines['bottom'].set_color('#e5e7eb')
    bar_ax.grid(True, linestyle='-', alpha=0.1, axis='x')
    bar_ax.tick_params(colors='#6b7280', labelsize=10)
    
    bar_fig.tight_layout()

    # Enhanced Line chart for emotion trends with smoothing
    line_fig, ax = plt.subplots(figsize=(12, 6))
    line_fig.patch.set_facecolor('#ffffff')

    # Ensure we have enough colors for all emotions
    colors = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444', '#06b6d4', '#84cc16']
    # Extend colors if we have more emotions than colors
    while len(colors) < len(emotions):
        colors.extend(colors)

    for i, emotion in enumerate(emotions):
        values_trend = [e['emotions'].get(emotion, 0.0) for e in emotion_history]
        # Apply Gaussian smoothing for better visualization
        smoothed = gaussian_filter1d(values_trend, sigma=1.5)
        color = colors[i % len(colors)]  # Use modulo to prevent index out of range
        ax.plot(smoothed, label=emotion, color=color, linewidth=2.5, marker='o', markersize=4)

    ax.set_title("Emotion Trends Throughout the Call", fontsize=16, fontweight='bold', color='#1f2937', pad=20)
    ax.set_xlabel("Time Segments", fontsize=12, color='#6b7280')
    ax.set_ylabel("Emotion Intensity", fontsize=12, color='#6b7280')
    
    # Enhanced legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=10)
    
    # Enhanced styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5e7eb')
    ax.spines['bottom'].set_color('#e5e7eb')
    ax.grid(True, linestyle='-', alpha=0.1)
    ax.tick_params(colors='#6b7280', labelsize=10)
    
    line_fig.tight_layout()

    return f"✅ Analysis completed successfully!\n{timing_info}", fig_to_image(bar_fig), fig_to_image(line_fig), full_transcript


def process_batch(audio_files):
    """Process multiple audio files in batch"""
    if not audio_files:
        return "❌ No files uploaded", [], [], "No transcriptions available.", "❌ Please upload audio files first."
    
    statuses = []
    bars = []
    lines = []
    transcripts = []
    
    # Process up to 10 files
    files_to_process = audio_files[:10] if len(audio_files) > 10 else audio_files
    
    for i, file in enumerate(files_to_process):
        print(f"Processing file {i+1}/{len(files_to_process)}: {file}")
        status, bar_fig, line_fig, transcript = run_live_analysis(file)
        
        # Format status with file info
        file_name = os.path.basename(file) if file else f"File {i+1}"
        formatted_status = f"📁 {file_name}\n{status}\n" + "="*50
        
        statuses.append(formatted_status)
        if bar_fig:
            bars.append(bar_fig)
        if line_fig:
            lines.append(line_fig)
        
        # Format transcript with file info
        formatted_transcript = f"📁 {file_name}:\n{transcript}\n" + "="*80 + "\n"
        transcripts.append(formatted_transcript)
    
    # Combine all results
    combined_status = "\n".join(statuses)
    combined_transcripts = "\n".join(transcripts)
    
    notification = f"✅ Successfully processed {len(files_to_process)} file(s)!"
    if len(audio_files) > 10:
        notification += f" (Limited to first 10 files, {len(audio_files) - 10} files skipped)"
    
    return combined_status, bars, lines, combined_transcripts, notification

# Enhanced Custom CSS for professional look
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #6366f1;
    --primary-hover: #5855eb;
    --secondary-color: #8b5cf6;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --text-primary: #1f2937;
    --text-secondary: #6b7280;
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --border-color: #e5e7eb;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
}

.gradio-container {
    background: transparent !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

/* Header Styling */
#header-section {
    background: var(--bg-primary);
    border-radius: 20px;
    padding: 3rem 2rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-xl);
    border: 1px solid var(--border-color);
    text-align: center;
    position: relative;
    overflow: hidden;
}

#header-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
}

#main-title {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin-bottom: 0.5rem !important;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

#subtitle {
    font-size: 1.125rem !important;
    color: var(--text-secondary) !important;
    line-height: 1.6 !important;
    max-width: 600px;
    margin: 0 auto;
}

/* Card Styling */
.card {
    background: var(--bg-primary) !important;
    border-radius: 16px !important;
    box-shadow: var(--shadow-lg) !important;
    border: 1px solid var(--border-color) !important;
    padding: 2rem !important;
    margin: 1rem 0 !important;
    transition: all 0.3s ease !important;
}

.card:hover {
    box-shadow: var(--shadow-xl) !important;
    transform: translateY(-2px) !important;
}

/* Input Section Styling */
#input-section {
    background: var(--bg-primary);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--border-color);
}

/* Button Styling */
#analyze-btn {
    background: linear-gradient(135deg, var(--primary-color), var(--primary-hover)) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.875rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: white !important;
    box-shadow: var(--shadow-md) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin-top: 1rem !important;
}

#analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg) !important;
    background: linear-gradient(135deg, var(--primary-hover), var(--primary-color)) !important;
}

#analyze-btn:active {
    transform: translateY(0) !important;
}

/* Audio Input Styling */
.gr-audio {
    border-radius: 12px !important;
    border: 2px dashed var(--border-color) !important;
    background: var(--bg-secondary) !important;
    transition: all 0.3s ease !important;
}

.gr-audio:hover {
    border-color: var(--primary-color) !important;
    background: var(--bg-primary) !important;
}

/* Instructions Styling */
#instructions {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe) !important;
    border-left: 4px solid var(--primary-color) !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
    margin-top: 1.5rem !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
}

#instructions strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* Results Section Styling */
#results-section {
    background: var(--bg-primary);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--border-color);
}

/* Status Output Styling */
.gr-textbox {
    border-radius: 8px !important;
    border: 1px solid var(--border-color) !important;
    font-weight: 500 !important;
}

/* Gallery Styling */
.gr-gallery {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-md) !important;
    border: 1px solid var(--border-color) !important;
}

/* Plot Styling */
.gr-plot {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-md) !important;
    border: 1px solid var(--border-color) !important;
}

/* Loading Animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s infinite;
}

/* Responsive Design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 1rem !important;
    }
    
    #main-title {
        font-size: 2rem !important;
    }
    
    #subtitle {
        font-size: 1rem !important;
    }
    
    .card {
        padding: 1.5rem !important;
    }
}

/* Feature Cards */
.feature-card {
    background: var(--bg-primary);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
}

.feature-icon {
    font-size: 2rem;
    margin-bottom: 1rem;
}

/* Batch Processing Styling */
.batch-info {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border-left: 4px solid var(--warning-color);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    font-size: 0.875rem;
}
"""

# Create the enhanced Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="AI Call Emotion Analysis") as demo:
    
    # Header Section
    with gr.Row(elem_id="header-section"):
        with gr.Column():
            gr.HTML("""
                <div style="text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🎙️</div>
                    <div id='main-title'>AI Call Emotion Analysis</div>
                    <div id='subtitle'>
                        Advanced sentiment analysis powered by AI. Upload your call recordings to get detailed 
                        emotional insights, trends, and comprehensive analytics with full transcriptions.
                    </div>
                </div>
            """)
    
    # Main Content
    with gr.Row():
        # Input Section
        with gr.Column(scale=1):
            with gr.Group(elem_id="input-section"):
                gr.Markdown("### 📁 Upload Audio Files", elem_id="section-title")
                
                # Multiple file upload with batch processing
                audio_input = gr.File(
                    file_count="multiple",
                    file_types=["audio"],
                    label="Upload up to 10 Call Recordings",
                    show_label=True,
                    interactive=True
                )
                
                analyze_btn = gr.Button(
                    "🚀 Analyze All Files", 
                    elem_id="analyze-btn", 
                    variant="primary",
                    size="lg"
                )
                
                gr.HTML("""
                    <div id='instructions'>
                        <strong>📋 Batch Processing Instructions:</strong><br>
                        • Upload multiple audio files (MP3, WAV, M4A)<br>
                        • Maximum 10 files per batch for optimal performance<br>
                        • Click "Analyze All Files" to process all uploads<br>
                        • View individual results and transcriptions below<br><br>
                        <strong>💡 Features:</strong><br>
                        • Gaussian smoothing for trend visualization<br>
                        • Full transcription for each audio file<br>
                        • Batch processing with progress tracking
                    </div>
                """)
        
        # Results Section  
        with gr.Column(scale=2):
            with gr.Group(elem_id="results-section"):
                gr.Markdown("### 📊 Batch Analysis Results", elem_id="section-title")
                
                # Notification for batch processing
                notification_out = gr.Textbox(
                    label="🔔 Processing Status", 
                    interactive=False,
                    placeholder="Ready to process your audio files...",
                    show_label=True
                )
                
                # Status for all files
                status_out = gr.Textbox(
                    label="📋 Detailed Status", 
                    interactive=False,
                    lines=8,
                    placeholder="Upload files and click analyze to see detailed status for each file...",
                    show_label=True
                )
    
    # Results Display Section
    with gr.Row():
        with gr.Column():
            # Gallery for bar charts
            bar_fig_out = gr.Gallery(
                label="📈 Emotion Distribution Charts",
                show_label=True,
                columns=2,
                rows=2,
                height="auto"
            )
    
    with gr.Row():
        with gr.Column():
            # Gallery for line charts  
            line_fig_out = gr.Gallery(
                label="📉 Emotion Timeline Charts", 
                show_label=True,
                columns=2,
                rows=2,
                height="auto"
            )
    
    # Transcription Section
    with gr.Row():
        with gr.Column():
            transcript_out = gr.Textbox(
                label="📝 Full Transcriptions", 
                interactive=False,
                lines=15,
                placeholder="Transcriptions will appear here after analysis...",
                show_label=True
            )
    
    # Features Section
    with gr.Row():
        gr.HTML("""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-top: 2rem;">
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <h3 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">Batch Processing</h3>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 0.875rem;">
                        Process up to 10 audio files simultaneously with detailed individual results.
                    </p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📝</div>
                    <h3 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">Full Transcription</h3>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 0.875rem;">
                        Get complete transcriptions alongside emotion analysis for each audio file.
                    </p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📈</div>
                    <h3 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">Smoothed Trends</h3>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 0.875rem;">
                        Gaussian smoothing applied to emotion trends for clearer visualization.
                    </p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔒</div>
                    <h3 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">Secure Processing</h3>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 0.875rem;">
                        All files processed securely with no permanent storage of sensitive data.
                    </p>
                </div>
            </div>
        """)

    # Event Handler for batch processing
    analyze_btn.click(
        process_batch, 
        inputs=audio_input, 
        outputs=[status_out, bar_fig_out, line_fig_out, transcript_out, notification_out],
        show_progress=True
    )

# Launch configuration
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # fallback to 8080 locally
    demo.launch(server_name="0.0.0.0", server_port=port , share=False, debug=True, inbrowser=True)
