// This is a diagnostic script to help identify issues with provider loading
console.log('Provider test script loaded');

// Function to check if provider sections exist and are accessible
function checkProviderSections() {
    console.log('Checking provider sections...');
    
    // Try different selectors to find the provider sections
    const selectors = [
        '.provider-section .collapsible-content',
        '.provider-section:nth-of-type(1) .collapsible-content',
        '.field.collapsible.provider-section .collapsible-content',
        '.settings-content .provider-section .collapsible-content',
        '.settings-content .provider-section:nth-of-type(1) .collapsible-content'
    ];
    
    selectors.forEach(selector => {
        const element = document.querySelector(selector);
        console.log(`Selector "${selector}": ${element ? 'Found' : 'Not found'}`);
        if (element) {
            console.log(`Element HTML: ${element.outerHTML.substring(0, 100)}...`);
        }
    });
    
    // Check API key section
    const apiKeySelectors = [
        '.collapsible-content.api-key',
        '.provider-section .collapsible-content.api-key',
        '.settings-content .collapsible-content.api-key'
    ];
    
    apiKeySelectors.forEach(selector => {
        const element = document.querySelector(selector);
        console.log(`API Key Selector "${selector}": ${element ? 'Found' : 'Not found'}`);
        if (element) {
            console.log(`Element HTML: ${element.outerHTML.substring(0, 100)}...`);
        }
    });
}

// Function to test API endpoint
async function testApiEndpoint() {
    console.log('Testing API endpoint...');
    try {
        const response = await fetch('/backend-api/v2/providers');
        if (!response.ok) {
            console.error(`Failed to fetch providers: ${response.status}`);
            return;
        }
        const providers = await response.json();
        console.log(`API returned ${providers.length} providers`);
        console.log('First 5 providers:', providers.slice(0, 5));
    } catch (error) {
        console.error('Error fetching providers:', error);
    }
}

// Function to manually add providers to the sections
async function manuallyAddProviders() {
    console.log('Manually adding providers...');
    try {
        const response = await fetch('/backend-api/v2/providers');
        if (!response.ok) {
            console.error(`Failed to fetch providers: ${response.status}`);
            return;
        }
        const providers = await response.json();
        
        // Sort providers alphabetically
        providers.sort((a, b) => a.label.localeCompare(b.label));
        
        // Get the provider sections
        const enableDisableSection = document.querySelector('.provider-section:nth-of-type(1) .collapsible-content');
        const apiKeySection = document.querySelector('.collapsible-content.api-key');
        
        if (!enableDisableSection) {
            console.error('Enable/Disable section not found');
            return;
        }
        
        if (!apiKeySection) {
            console.error('API Key section not found');
            return;
        }
        
        // Clear existing providers
        enableDisableSection.innerHTML = '';
        apiKeySection.innerHTML = '';
        
        // Add providers to Enable/Disable section
        let enabledCount = 0;
        providers.forEach(provider => {
            if (!provider.parent) {
                const providerItem = document.createElement('div');
                providerItem.classList.add('provider-item');
                
                const apiKey = localStorage.getItem(`${provider.name}-api_key`);
                
                providerItem.innerHTML = `
                    <span class="label">${provider.label}</span>
                    <input type="checkbox" id="provider-${provider.name.toLowerCase()}" ${!provider.auth || apiKey ? 'checked="checked"' : ''}/>
                    <label for="provider-${provider.name.toLowerCase()}" class="toogle" title="Enable/Disable ${provider.label}"></label>
                `;
                
                enableDisableSection.appendChild(providerItem);
                enabledCount++;
            }
        });
        console.log(`Added ${enabledCount} providers to Enable/Disable section`);
        
        // Add providers to API Key section
        let apiKeyCount = 0;
        providers.forEach(provider => {
            if (provider.auth) {
                const apiKeyItem = document.createElement('div');
                apiKeyItem.classList.add('field', 'box');
                
                apiKeyItem.innerHTML = `
                    <label for="${provider.name}-api_key" class="label">${provider.label} API Key:</label>
                    <input type="text" id="${provider.name}-api_key" name="${provider.name}[api_key]" placeholder="API Key" />
                    ${provider.login_url ? `<a href="${provider.login_url}" target="_blank">Get API Key</a>` : ''}
                `;
                
                apiKeySection.appendChild(apiKeyItem);
                apiKeyCount++;
            }
        });
        console.log(`Added ${apiKeyCount} providers to API Key section`);
        
        // Reinitialize collapsible sections
        if (window.setupCollapsibleFields) {
            console.log('Reinitializing collapsible fields');
            window.setupCollapsibleFields();
        } else {
            console.warn('setupCollapsibleFields function not found');
        }
    } catch (error) {
        console.error('Error manually adding providers:', error);
    }
}

// Run tests when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM content loaded, running tests...');
    
    // Add a button to the settings panel to run the tests
    const settingsContent = document.querySelector('.settings-content');
    if (settingsContent) {
        const testButton = document.createElement('button');
        testButton.innerHTML = '<i class="fa-solid fa-vial"></i><span>Run Provider Tests</span>';
        testButton.style.marginTop = '20px';
        testButton.addEventListener('click', function() {
            checkProviderSections();
            testApiEndpoint();
            manuallyAddProviders();
        });
        
        settingsContent.appendChild(testButton);
    }
    
    // Also run when the settings panel is opened
    const settingsIcon = document.querySelector('.settings_icon');
    if (settingsIcon) {
        settingsIcon.addEventListener('click', function() {
            setTimeout(function() {
                const testButton = document.querySelector('.settings-content button:last-child');
                if (!testButton || !testButton.innerHTML.includes('Run Provider Tests')) {
                    const settingsContent = document.querySelector('.settings-content');
                    if (settingsContent) {
                        const testButton = document.createElement('button');
                        testButton.innerHTML = '<i class="fa-solid fa-vial"></i><span>Run Provider Tests</span>';
                        testButton.style.marginTop = '20px';
                        testButton.addEventListener('click', function() {
                            checkProviderSections();
                            testApiEndpoint();
                            manuallyAddProviders();
                        });
                        
                        settingsContent.appendChild(testButton);
                    }
                }
            }, 1000);
        });
    }
});
