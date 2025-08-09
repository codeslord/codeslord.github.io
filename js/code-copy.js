/* Code Copy Button Functionality */
(function() {
    'use strict';
    
    // Global flag to prevent duplicate initialization
    let copyButtonsInitialized = false;
    
    // Add copy buttons to all code blocks when DOM is loaded
    function addCopyButtons() {
        const codeBlocks = document.querySelectorAll('.highlight');
        
        codeBlocks.forEach(function(codeBlock) {
            // Skip if copy button already exists - check more thoroughly
            const existingButtons = codeBlock.querySelectorAll('.code-copy-btn');
            if (existingButtons.length > 0) {
                // Remove duplicate buttons if they exist
                for (let i = 1; i < existingButtons.length; i++) {
                    existingButtons[i].remove();
                }
                return;
            }
            
            // Skip nested highlight blocks to prevent duplicates
            if (codeBlock.closest('.highlight') !== codeBlock) {
                return;
            }
            
            // Additional check: skip if parent already has a copy button
            const parentHighlight = codeBlock.parentElement.closest('.highlight');
            if (parentHighlight && parentHighlight !== codeBlock && parentHighlight.querySelector('.code-copy-btn')) {
                return;
            }
            
            const copyButton = document.createElement('button');
            copyButton.className = 'code-copy-btn';
            copyButton.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
            copyButton.setAttribute('aria-label', 'Copy code to clipboard');
            copyButton.setAttribute('title', 'Copy code');
            
            // Add click event listener
            copyButton.addEventListener('click', function() {
                copyCodeToClipboard(codeBlock, copyButton);
            });
            
            codeBlock.appendChild(copyButton);
        });
    }
    
    // Copy code content to clipboard
    function copyCodeToClipboard(codeBlock, button) {
        const preElement = codeBlock.querySelector('pre');
        if (!preElement) return;
        
        // Clone the pre element to avoid modifying the original
        const preClone = preElement.cloneNode(true);
        
        // Remove any copy buttons from the clone
        const copyButtons = preClone.querySelectorAll('.code-copy-btn');
        copyButtons.forEach(function(btn) {
            btn.remove();
        });
        
        // Get the text content, preserving line breaks
        let codeText = preClone.textContent || preClone.innerText;
        
        // Remove any leading/trailing whitespace
        codeText = codeText.trim();
        
        // Use modern clipboard API if available
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(codeText).then(function() {
                showCopySuccess(button);
            }).catch(function(err) {
                console.error('Failed to copy code: ', err);
                fallbackCopyTextToClipboard(codeText, button);
            });
        } else {
            // Fallback for older browsers
            fallbackCopyTextToClipboard(codeText, button);
        }
    }
    
    // Fallback copy method for older browsers
    function fallbackCopyTextToClipboard(text, button) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        
        // Make the textarea out of viewport
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            const successful = document.execCommand('copy');
            if (successful) {
                showCopySuccess(button);
            } else {
                console.error('Fallback: Copy command was unsuccessful');
            }
        } catch (err) {
            console.error('Fallback: Unable to copy', err);
        }
        
        document.body.removeChild(textArea);
    }
    
    // Show visual feedback when copy is successful
    function showCopySuccess(button) {
        const originalHTML = button.innerHTML;
        button.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 12 2 2 4-4"></path></svg>';
        button.classList.add('copied');
        
        setTimeout(function() {
            button.innerHTML = originalHTML;
            button.classList.remove('copied');
        }, 2000);
    }
    
    // Initialize when DOM is loaded
    function initializeCopyButtons() {
        if (copyButtonsInitialized) return;
        copyButtonsInitialized = true;
        addCopyButtons();
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeCopyButtons);
    } else {
        initializeCopyButtons();
    }
    
    // Re-run only for genuinely new content, not theme changes
    const observer = new MutationObserver(function(mutations) {
        let hasNewHighlightBlocks = false;
        
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType === 1) { // Element node
                        if (node.classList && node.classList.contains('highlight') && !node.querySelector('.code-copy-btn')) {
                            hasNewHighlightBlocks = true;
                        } else if (node.querySelector) {
                            const newHighlights = node.querySelectorAll('.highlight');
                            newHighlights.forEach(function(highlight) {
                                if (!highlight.querySelector('.code-copy-btn')) {
                                    hasNewHighlightBlocks = true;
                                }
                            });
                        }
                    }
                });
            }
        });
        
        if (hasNewHighlightBlocks) {
            addCopyButtons();
        }
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
})();
