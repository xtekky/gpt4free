// Function to dynamically load all providers into the Enable/Disable and API key sections
async function loadAllProviders() {
    try {
        // Fetch providers from the API
        const providers = await fetchProviders();
        if (!providers || providers.length === 0) {
            return; // No providers found
        }

        // Sort providers alphabetically
        providers.sort((a, b) => a.label.localeCompare(b.label));

        // Get the provider sections using the unique IDs we added
        const enableDisableSection = document.querySelector('#enable-disable-section .collapsible-content');
        const apiKeySection = document.querySelector('#api-key-section .collapsible-content');

        if (!enableDisableSection || !apiKeySection) {
            return; // One or both provider sections not found
        }

        // Clear existing hardcoded providers
        enableDisableSection.innerHTML = '';
        apiKeySection.innerHTML = '';

        // Add all providers to the Enable/Disable section
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
            }
        });

        // Add providers that need API keys to the API key section
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
            }
        });

        // Reinitialize collapsible sections
        if (window.setupCollapsibleFields) {
            window.setupCollapsibleFields();
        }
    } catch (error) {
        // Handle error silently
    }
}

// Function to fetch providers from the API
async function fetchProviders() {
    try {
        const response = await fetch('/backend-api/v2/providers');
        if (!response.ok) {
            throw new Error(`Failed to fetch providers: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        return []; // Return empty array on error
    }
}

// Load providers when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(loadAllProviders, 1000); // Increased timeout to 1000ms
});

// Also try loading when the settings panel is opened
document.addEventListener('DOMContentLoaded', function() {
    const settingsIcon = document.querySelector('.settings_icon');
    if (settingsIcon) {
        settingsIcon.addEventListener('click', function() {
            setTimeout(loadAllProviders, 500);
        });
    }
});
