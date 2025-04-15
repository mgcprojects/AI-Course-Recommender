\# Venv and VS Code

\#\# Set Up a Python Environment (Recommended):

Using a virtual environment keeps dependencies for this project separate from your global Python installation.

1\.  Open a terminal within VS Code: \`Terminal \> New Terminal\`.  
2\.  In the terminal, create a virtual environment (ex: \`.venv\`):

    \`\`\`bash  
    python \-m venv .venv  
    \`\`\`

    (If \`python\` doesn't work, try \`py \-m venv .venv\`)

3\.  Activate the virtual environment:

    \`\`\`bash  
    .\\.venv\\Scripts\\activate  
    \`\`\`

    You should see \`(.venv)\` appear at the beginning of your terminal prompt, indicating it's active.  
4\.  Select the Interpreter: VS Code might automatically detect and ask if you want to use the interpreter in the new \`.venv\` folder. If so, click "Yes". If not, click on the Python version shown in the bottom status bar (or press \`Ctrl+Shift+P\` and type "Python: Select Interpreter") and choose the Python interpreter located inside the \`.venv\` folder (it will usually list the path like \`.\\.venv\\Scripts\\python.exe\`).

\#\# Install Required Libraries:

1\.  Make sure your virtual environment is still active (you see \`(.venv)\` in the terminal prompt).  
2\.  Install the necessary libraries using \`pip\`:

    \`\`\`bash  
    pip install requests nltk scikit-learn  
    \`\`\`

3\.  Wait for the installations to complete.

\#\# Run the Python Script:

You have several ways to run it:

\* \*\*Method 1 (Terminal):\*\* Ensure your virtual environment is active in the VS Code terminal. Type:

    \`\`\`bash  
    python Microsoft\_AIsearch.py  
    \`\`\`

\* \*\*Method 2 (Right-Click):\*\* Right-click anywhere inside the code editor for \`Microsoft\_AIsearch.py\` and select "Run Python File in Terminal".  
\* \*\*Method 3 (Run Button):\*\* Look for a green "Play" button (Run Python File) in the top-right corner of the VS Code window when the \`.py\` file is open. Click it.  
