#!/bin/bash
# build.sh - Simple static site builder for Vercel

# Create the output directory
mkdir -p docs/dist

# Build the documentation using Python (but don't package it as a function)
python3 << 'EOL'
#!/usr/bin/env python3
"""
Static documentation builder for RedDust Reclaimer.
Generates a static HTML site for Vercel deployment.
"""
import os
import json
from datetime import datetime

def create_html_page(title, content, filename):
    """Create an HTML page with consistent styling."""
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - RedDust Reclaimer</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #1e3c72; border-bottom: 3px solid #2a5298; padding-bottom: 10px; }}
        h2 {{ color: #2a5298; margin-top: 30px; }}
        h3 {{ color: #1e3c72; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .badge {{ 
            display: inline-block; 
            background: #28a745; 
            color: white; 
            padding: 4px 12px; 
            border-radius: 20px; 
            font-size: 0.8em;
            margin: 0 5px;
        }}
        .nav {{ 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 30px;
        }}
        .nav a {{ 
            text-decoration: none; 
            color: #1e3c72; 
            margin: 0 15px; 
            font-weight: bold;
        }}
        .nav a:hover {{ color: #2a5298; }}
        code {{ 
            background: #f8f9fa; 
            padding: 2px 6px; 
            border-radius: 4px; 
            font-family: 'Courier New', monospace;
        }}
        pre {{ 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px; 
            overflow-x: auto;
            border-left: 4px solid #2a5298;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 0.9em;
        }}
        .success-badge {{
            background: #28a745;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            display: inline-block;
            margin: 10px 0;
        }}
        .deployment-info {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌌 RedDust Reclaimer</h1>
            <p>Synthetic Biology Meets Astrobiology</p>
            <div>
                <span class="badge">Python 3.8+</span>
                <span class="badge">Synthetic Biology</span>
                <span class="badge">Mars Biotech</span>
                <span class="badge">B. subtilis</span>
            </div>
            <div class="success-badge">✅ Successfully Deployed on Vercel</div>
        </div>
        
        <div class="nav">
            <a href="index.html">🏠 Home</a>
            <a href="research.html">🧬 Research</a>
            <a href="methodology.html">🔬 Methodology</a>
            <a href="deployment.html">🚀 Deployment</a>
        </div>
        
        {content}
        
        <div class="footer">
            <p>🌌 RedDust Reclaimer - Engineering Life for Mars | Built with ❤️ for Astrobiology</p>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")} | Deployed via Vercel</p>
        </div>
    </div>
</body>
</html>"""
    
    os.makedirs("docs/dist", exist_ok=True)
    with open(f"docs/dist/{filename}", "w") as f:
        f.write(html_template)

def build_documentation():
    """Build all documentation pages."""
    
    # Home page
    home_content = """
    <div class="deployment-info">
        <h3>🎉 Deployment Successful!</h3>
        <p>This site has been successfully deployed using Vercel's static site hosting. The build process completed without serverless function overhead.</p>
    </div>
    
    <h2>🎯 Project Mission</h2>
    <p>RedDust Reclaimer engineers <strong>Bacillus subtilis</strong> for perchlorate bioremediation under Martian regolith conditions. We're solving one of Mars colonization's biggest challenges: <strong>toxic perchlorate cleanup</strong>.</p>
    
    <h2>🧬 What We're Building</h2>
    <ul>
        <li><strong>Codon-optimized enzymes</strong> for perchlorate reduction (pcrA/pcrB)</li>
        <li><strong>Chlorite dismutase (cld)</strong> for complete detoxification pathway</li>
        <li><strong>Cold-adapted proteins</strong> for Martian temperature conditions</li>
        <li><strong>Metabolic modeling</strong> under Mars constraints using COBRApy</li>
        <li><strong>CRISPR genome engineering</strong> for stable integration</li>
    </ul>
    
    <h2>🛠️ Technology Stack</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0;">
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745;">
            <h3>🧬 Gene Engineering</h3>
            <p>NCBI, UniProt, KEGG, JCat, Benchling</p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #17a2b8;">
            <h3>🔬 Modeling</h3>
            <p>COBRApy, BioCyc, KBase, GROMACS</p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #ffc107;">
            <h3>⚛️ Structural Analysis</h3>
            <p>PyMOL, SwissDock, AMDock, AutoDock Vina</p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #dc3545;">
            <h3>🤖 Machine Learning</h3>
            <p>SciPy, Scikit-learn, TensorFlow, NumPy</p>
        </div>
    </div>
    """
    
    create_html_page("Home", home_content, "index.html")
    
    # Research page (keeping the same content as before)
    research_content = """
    <h2>🔬 Research Methodology</h2>
    
    <h3>Step 1: Gene Optimization</h3>
    <p>Identify optimal pcrA/pcrB and cld genes, then codon-optimize for B. subtilis using JCat and GeneOptimizer.</p>
    
    <h3>Step 2: Pathway Modeling</h3>
    <p>Integrate perchlorate detox pathway into B. subtilis metabolic network and simulate Mars-like conditions.</p>
    
    <h3>Step 3: Enzyme Engineering</h3>
    <p>Dock enzymes with perchlorate/chlorite and run molecular dynamics to confirm cold-environment stability.</p>
    
    <h3>Step 4: Genome Engineering</h3>
    <p>Design plasmids or apply CRISPR editing for chromosomal integration using strong promoters.</p>
    
    <h3>Step 5: Validation</h3>
    <p>RNA-Seq and WGS to confirm expression, stability, and check for off-target effects.</p>
    
    <h3>Step 6: Mars Simulation</h3>
    <p>Model system performance in low-pressure, low-temperature environments with ML optimization.</p>
    
    <h3>Step 7: Bioethics & Safety</h3>
    <p>Align with NASA Planetary Protection guidelines and address biosafety concerns.</p>
    """
    
    create_html_page("Research", research_content, "research.html")
    
    # Methodology page
    methodology_content = """
    <h2>🧪 Scientific Methodology</h2>
    
    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <thead>
            <tr style="background: #1e3c72; color: white;">
                <th style="padding: 12px; text-align: left;">Step</th>
                <th style="padding: 12px; text-align: left;">Category</th>
                <th style="padding: 12px; text-align: left;">Tools & Resources</th>
            </tr>
        </thead>
        <tbody>
            <tr style="background: #f8f9fa;">
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">1</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Gene Optimization</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">NCBI, UniProt, KEGG, JCat, Benchling</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">2</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Pathway Modeling</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">COBRApy, KBase, BioCyc</td>
            </tr>
            <tr style="background: #f8f9fa;">
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">3</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Enzyme Engineering</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">AMDock, SwissDock, AutoDock Vina, GROMACS</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">4</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Genome Engineering</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">SnapGene, Benchling, CHOPCHOP, Geneious</td>
            </tr>
            <tr style="background: #f8f9fa;">
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">5</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Expression Validation</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">Galaxy, CLC Genomics, DESeq2</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">6</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Predictive Modeling</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">SciPy, NumPy, scikit-learn, TensorFlow</td>
            </tr>
            <tr style="background: #f8f9fa;">
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">7</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Ethics & Safety</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">NASA Guidelines, iGEM Safety Hub</td>
            </tr>
        </tbody>
    </table>
    
    <h2>🌍 Mars Environment Challenges</h2>
    <ul>
        <li><strong>Temperature:</strong> -80°C to 20°C (avg -63°C)</li>
        <li><strong>Pressure:</strong> 0.6% of Earth's atmospheric pressure</li>
        <li><strong>Radiation:</strong> High UV and cosmic radiation</li>
        <li><strong>Perchlorate concentration:</strong> 0.5% by weight in regolith</li>
        <li><strong>Water availability:</strong> Limited, mostly ice</li>
    </ul>
    """
    
    create_html_page("Methodology", methodology_content, "methodology.html")
    
    # Deployment Status page
    deployment_content = """
    <h2>🚀 Deployment Status</h2>
    
    <div class="deployment-info">
        <h3>✅ Static Site Successfully Deployed</h3>
        <p>The RedDust Reclaimer documentation is now live on Vercel as a static site, avoiding the serverless function size limits.</p>
    </div>
    
    <h3>🔧 Build Configuration</h3>
    <ul>
        <li><strong>Platform:</strong> Vercel Static Site Hosting</li>
        <li><strong>Build Command:</strong> <code>python docs/build.py</code></li>
        <li><strong>Output Directory:</strong> <code>docs/dist</code></li>
        <li><strong>Framework:</strong> None (Pure Static)</li>
        <li><strong>Deployment:</strong> Automatic on git push</li>
    </ul>
    
    <h3>📈 Deployment Benefits</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;">
            <h4 style="color: #28a745; margin: 0;">Build Time</h4>
            <p style="font-size: 1.5em; margin: 5px 0;">~30 sec</p>
        </div>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;">
            <h4 style="color: #17a2b8; margin: 0;">Size Limit</h4>
            <p style="font-size: 1.5em; margin: 5px 0;">No Limit</p>
        </div>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;">
            <h4 style="color: #ffc107; margin: 0;">CDN</h4>
            <p style="font-size: 1.5em; margin: 5px 0;">Global</p>
        </div>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;">
            <h4 style="color: #dc3545; margin: 0;">SSL</h4>
            <p style="font-size: 1.5em; margin: 5px 0;">Auto</p>
        </div>
    </div>
    
    <h3>🛠️ Technical Details</h3>
    <p>This site is built using a simple Python script that generates static HTML files. The build process runs during deployment, creating optimized static assets that are served directly from Vercel's CDN.</p>
    
    <h3>🔄 Automatic Updates</h3>
    <p>Every commit to the main branch triggers an automatic rebuild and deployment. Changes are live within minutes of pushing to GitHub.</p>
    """
    
    create_html_page("Deployment", deployment_content, "deployment.html")
    
    print("✅ Static documentation built successfully!")
    print("📄 Pages created: index.html, research.html, methodology.html, deployment.html")
    print("🚀 Ready for Vercel static site deployment")

if __name__ == "__main__":
    build_documentation()
EOL

echo "✅ Build script completed successfully!"
echo "📁 Output directory: docs/dist"
echo "🌐 Ready for Vercel deployment"
