// Function to dynamically load all providers into the Enable/Disable and API key sections
async function loadAllProviders() {
    try {
        console.log('Starting to load providers...');
        
        // Fetch providers from the API
        const providers = await fetchProviders();
        if (!providers || providers.length === 0) {
            console.error('No providers found');
            return;
        }
        
        console.log(`Fetched ${providers.length} providers from API`);

        // Sort providers alphabetically
        providers.sort((a, b) => a.label.localeCompare(b.label));

        // Get the provider sections using the unique IDs we added
        let enableDisableSection = document.querySelector('#enable-disable-section .collapsible-content');
        if (!enableDisableSection) {
            console.log('ID selector failed, trying alternative selector for enable/disable section');
            enableDisableSection = document.querySelector('.provider-section:nth-of-type(1) .collapsible-content');
            if (!enableDisableSection) {
                console.log('Second selector failed, trying generic selector');
                enableDisableSection = document.querySelector('.provider-section .collapsible-content');
            }
        }
        
        let apiKeySection = document.querySelector('#api-key-section .collapsible-content');
        if (!apiKeySection) {
            console.log('ID selector failed, trying alternative selector for API key section');
            apiKeySection = document.querySelector('.collapsible-content.api-key');
        }
        
        // Log what we found
        console.log('Enable/Disable section found:', !!enableDisableSection);
        console.log('API Key section found:', !!apiKeySection);

        if (!enableDisableSection) {
            console.error('Enable/Disable section not found. Dumping all .provider-section elements:');
            const providerSections = document.querySelectorAll('.provider-section');
            console.log(`Found ${providerSections.length} .provider-section elements`);
            providerSections.forEach((section, i) => {
                console.log(`Section ${i}:`, section.outerHTML);
            });
        }

        if (!apiKeySection) {
            console.error('API Key section not found. Dumping all .collapsible-content elements:');
            const collapsibleContents = document.querySelectorAll('.collapsible-content');
            console.log(`Found ${collapsibleContents.length} .collapsible-content elements`);
            collapsibleContents.forEach((content, i) => {
                console.log(`Content ${i}:`, content.outerHTML);
            });
        }

        if (!enableDisableSection || !apiKeySection) {
            console.error('One or both provider sections not found');
            return;
        }

        // Clear existing hardcoded providers
        console.log('Clearing existing providers');
        enableDisableSection.innerHTML = '';
        apiKeySection.innerHTML = '';

        // Add all providers to the Enable/Disable section
        console.log('Adding providers to Enable/Disable section');
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

        // Add providers that need API keys to the API key section
        console.log('Adding providers to API Key section');
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
        
        console.log(`Successfully loaded ${providers.length} providers`);
    } catch (error) {
        console.error('Error loading providers:', error);
    }
}

// Function to fetch providers from the API
async function fetchProviders() {
    try {
        console.log('Fetching providers from API...');
        const response = await fetch('/backend-api/v2/providers');
        if (!response.ok) {
            throw new Error(`Failed to fetch providers: ${response.status}`);
        }
        const data = await response.json();
        console.log(`Successfully fetched ${data.length} providers from API`);
        return data;
    } catch (error) {
        console.error('Error fetching providers:', error);
        return [];
    }
}

// Load providers when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM content loaded, scheduling provider loading');
    // Wait a short time to ensure the main JS has initialized
    setTimeout(() => {
        console.log('Executing loadAllProviders after timeout');
        loadAllProviders();
    }, 1000); // Increased timeout to 1000ms
});

// Also try loading when the settings panel is opened
document.addEventListener('DOMContentLoaded', function() {
    const settingsIcon = document.querySelector('.settings_icon');
    if (settingsIcon) {
        console.log('Adding click listener to settings icon');
        settingsIcon.addEventListener('click', function() {
            console.log('Settings icon clicked, loading providers');
            setTimeout(loadAllProviders, 500);
        });
    } else {
        console.warn('Settings icon not found');
    }
});
