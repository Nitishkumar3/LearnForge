/**
 * LearnForge Navbar Module
 * Handles: Auth, Theme, User Menu, Workspace Switcher
 *
 * Usage:
 *   1. Include this script before page-specific scripts
 *   2. Call Navbar.init() to load user and workspaces
 *   3. Register workspace switch callback: Navbar.onWorkspaceSwitch(callback)
 */

// ===========================================
// AUTH UTILITIES (Global for backward compat)
// ===========================================
const Auth = {
    TOKEN_KEY: 'learnforge_token',
    USER_KEY: 'learnforge_user',
    WORKSPACE_KEY: 'learnforge_workspace',

    getToken() {
        return localStorage.getItem(this.TOKEN_KEY);
    },

    getUser() {
        const user = localStorage.getItem(this.USER_KEY);
        return user ? JSON.parse(user) : null;
    },

    getCurrentWorkspace() {
        const ws = localStorage.getItem(this.WORKSPACE_KEY);
        return ws ? JSON.parse(ws) : null;
    },

    setCurrentWorkspace(workspace) {
        localStorage.setItem(this.WORKSPACE_KEY, JSON.stringify(workspace));
    },

    isAuthenticated() {
        return !!this.getToken();
    },

    logout() {
        localStorage.removeItem(this.TOKEN_KEY);
        localStorage.removeItem(this.USER_KEY);
        localStorage.removeItem(this.WORKSPACE_KEY);
        window.location.href = '/signin';
    },

    getAuthHeaders() {
        const token = this.getToken();
        return token ? { 'Authorization': `Bearer ${token}` } : {};
    }
};

// ===========================================
// UTILITY FUNCTIONS (Global for backward compat)
// ===========================================
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `fixed bottom-4 right-4 bg-[#0D0D0D] dark:bg-[#ECECEC] text-white dark:text-[#0D0D0D] px-4 py-3 rounded-xl shadow-lg z-50 text-sm font-medium transition-all duration-300`;
    toast.style.animation = 'fadeInUp 0.3s ease-out';
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'fadeOutDown 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, 2700);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===========================================
// STATE (Global for backward compat)
// ===========================================
let workspaces = [];
let currentWorkspace = null;
let currentUser = null;

// ===========================================
// THEME FUNCTIONS
// ===========================================
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.classList.contains('dark') ? 'dark' : 'light';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    html.classList.remove('light', 'dark');
    html.classList.add(newTheme);
    localStorage.setItem('learnforge_theme', newTheme);

    // Update highlight.js theme if present
    const lightStyle = document.getElementById('hljs-light');
    const darkStyle = document.getElementById('hljs-dark');
    if (lightStyle && darkStyle) {
        if (newTheme === 'dark') {
            lightStyle.disabled = true;
            darkStyle.disabled = false;
        } else {
            lightStyle.disabled = false;
            darkStyle.disabled = true;
        }
    }
}

function initializeTheme() {
    const savedTheme = localStorage.getItem('learnforge_theme') || 'light';
    const lightStyle = document.getElementById('hljs-light');
    const darkStyle = document.getElementById('hljs-dark');
    if (lightStyle && darkStyle) {
        if (savedTheme === 'dark') {
            lightStyle.disabled = true;
            darkStyle.disabled = false;
        } else {
            lightStyle.disabled = false;
            darkStyle.disabled = true;
        }
    }
}

// Initialize theme immediately
initializeTheme();

// ===========================================
// NAVBAR MODULE
// ===========================================
const Navbar = (function() {
    // Private: Callback for workspace switch
    let _onWorkspaceSwitchCallback = null;

    // Update user display in navbar
    function updateUserDisplay(user) {
        currentUser = user;
        const userNameEl = document.getElementById('userName');
        const userEmailEl = document.getElementById('userEmail');
        const userInitialEl = document.getElementById('userInitial');

        if (userNameEl) userNameEl.textContent = user.name;
        if (userEmailEl) userEmailEl.textContent = user.email;
        if (userInitialEl) userInitialEl.textContent = user.name.charAt(0).toUpperCase();
    }

    // Update workspace display in navbar
    function updateWorkspaceDisplay() {
        const wsNameEl = document.getElementById('currentWorkspaceName');
        const wsDropdownEl = document.getElementById('workspaceDropdown');

        if (wsNameEl && currentWorkspace) {
            wsNameEl.textContent = currentWorkspace.name;
        }

        if (wsDropdownEl) {
            wsDropdownEl.innerHTML = workspaces.map(ws => `
                <div class="workspace-item flex items-center justify-between p-2 rounded-lg hover:bg-light-hover dark:hover:bg-dark-hover cursor-pointer group ${ws.id === currentWorkspace?.id ? 'bg-light-hover dark:bg-dark-hover' : ''}"
                     onclick="Navbar.switchWorkspace('${ws.id}')">
                    <div class="flex items-center gap-2 min-w-0 flex-1">
                        <div class="w-6 h-6 rounded-md bg-light-text dark:bg-dark-text flex items-center justify-center text-xs font-semibold text-light-bg dark:text-dark-bg flex-shrink-0">
                            ${ws.name.charAt(0).toUpperCase()}
                        </div>
                        <span class="text-sm truncate ${ws.id === currentWorkspace?.id ? 'font-medium text-light-text dark:text-dark-text' : 'text-light-text-secondary dark:text-dark-text-secondary'}">${escapeHtml(ws.name)}</span>
                    </div>
                    <div class="flex items-center gap-1 flex-shrink-0">
                        ${ws.id === currentWorkspace?.id ? '<svg class="w-4 h-4 text-light-text dark:text-dark-text" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>' : ''}
                        <button onclick="Navbar.openWorkspaceSettings('${ws.id}', event)"
                                class="p-1 text-light-text-muted dark:text-dark-text-muted hover:text-light-text dark:hover:text-dark-text hover:bg-light-hover dark:hover:bg-dark-hover rounded opacity-0 group-hover:opacity-100 transition-opacity"
                                title="Workspace settings">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                            </svg>
                        </button>
                    </div>
                </div>
            `).join('');
        }

        // Update navigation links with workspace_id
        if (currentWorkspace) {
            const studyLink = document.getElementById('studyMaterialsLink');
            const quizLink = document.getElementById('quizBtn');
            if (studyLink) studyLink.href = '/study/' + currentWorkspace.id;
            if (quizLink) quizLink.href = '/quiz/' + currentWorkspace.id;
        }
    }

    // Toggle workspace dropdown
    function toggleWorkspaceDropdown() {
        const dropdown = document.getElementById('workspaceDropdownContainer');
        if (dropdown) dropdown.classList.toggle('hidden');
    }

    // Toggle user dropdown
    function toggleUserDropdown() {
        const dropdown = document.getElementById('userDropdownContainer');
        if (dropdown) dropdown.classList.toggle('hidden');
    }

    // Switch workspace
    async function switchWorkspace(workspaceId, updateUrl = true) {
        const workspace = workspaces.find(w => w.id === workspaceId);
        if (workspace && workspace.id !== currentWorkspace?.id) {
            currentWorkspace = workspace;
            Auth.setCurrentWorkspace(workspace);
            updateWorkspaceDisplay();
            toggleWorkspaceDropdown();

            // Call page-specific callback if registered
            if (_onWorkspaceSwitchCallback) {
                await _onWorkspaceSwitchCallback(workspace, updateUrl);
            }

            showToast(`Switched to ${workspace.name}`, 'success');
        }
    }

    // Open create workspace modal
    function openCreateWorkspaceModal() {
        document.getElementById('createWorkspaceModal').classList.remove('hidden');
        document.getElementById('newWorkspaceName').value = '';
        document.getElementById('newWorkspaceName').focus();
        // Close workspace dropdown
        document.getElementById('workspaceDropdownContainer').classList.add('hidden');
    }

    // Close create workspace modal
    function closeCreateWorkspaceModal() {
        document.getElementById('createWorkspaceModal').classList.add('hidden');
    }

    // Submit create workspace
    async function submitCreateWorkspace() {
        const name = document.getElementById('newWorkspaceName').value.trim();
        if (!name) {
            showToast('Please enter a workspace name', 'error');
            return;
        }

        try {
            const response = await fetch('/api/workspaces', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...Auth.getAuthHeaders()
                },
                body: JSON.stringify({ name })
            });

            if (response.ok) {
                const data = await response.json();
                workspaces.push(data.workspace);
                closeCreateWorkspaceModal();
                await switchWorkspace(data.workspace.id);
                showToast('Workspace created', 'success');
            } else {
                const error = await response.json();
                showToast(error.error || 'Failed to create workspace', 'error');
            }
        } catch (error) {
            showToast('Failed to create workspace', 'error');
        }
    }

    // Open workspace settings modal
    function openWorkspaceSettings(workspaceId, event) {
        if (event) event.stopPropagation();

        const workspace = workspaces.find(w => w.id === workspaceId);
        if (!workspace) return;

        document.getElementById('settingsWorkspaceId').value = workspaceId;
        document.getElementById('editWorkspaceName').value = workspace.name;
        document.getElementById('workspaceSettingsModal').classList.remove('hidden');
        // Close workspace dropdown
        document.getElementById('workspaceDropdownContainer').classList.add('hidden');

        // Call page-specific syllabus update if function exists
        if (typeof updateSyllabusView === 'function') {
            updateSyllabusView();
        }
    }

    // Initialize: Load user and workspaces
    async function init() {
        try {
            // Load user info
            const userResponse = await fetch('/api/auth/me', {
                headers: Auth.getAuthHeaders()
            });

            if (!userResponse.ok) {
                Auth.logout();
                return;
            }

            const userData = await userResponse.json();
            updateUserDisplay(userData.user);

            // Load workspaces
            const wsResponse = await fetch('/api/workspaces', {
                headers: Auth.getAuthHeaders()
            });

            if (wsResponse.ok) {
                const wsData = await wsResponse.json();
                workspaces = wsData.workspaces;

                // Set current workspace
                const savedWorkspace = Auth.getCurrentWorkspace();
                if (savedWorkspace && workspaces.find(w => w.id === savedWorkspace.id)) {
                    currentWorkspace = savedWorkspace;
                } else if (workspaces.length > 0) {
                    currentWorkspace = workspaces[0];
                    Auth.setCurrentWorkspace(currentWorkspace);
                }

                updateWorkspaceDisplay();
            }

            return { user: currentUser, workspace: currentWorkspace };
        } catch (error) {
            console.error('Error loading user/workspaces:', error);
            return null;
        }
    }

    // Setup click outside handlers for dropdowns
    function setupClickOutsideHandlers() {
        document.addEventListener('click', (e) => {
            const wsDropdown = document.getElementById('workspaceDropdownContainer');
            const wsButton = document.getElementById('workspaceSwitcher');
            const userDropdown = document.getElementById('userDropdownContainer');
            const userButton = document.getElementById('userMenuButton');

            if (wsDropdown && !wsDropdown.contains(e.target) && !wsButton?.contains(e.target)) {
                wsDropdown.classList.add('hidden');
            }
            if (userDropdown && !userDropdown.contains(e.target) && !userButton?.contains(e.target)) {
                userDropdown.classList.add('hidden');
            }
        });
    }

    // Setup handlers immediately
    setupClickOutsideHandlers();

    // Public API
    return {
        init,
        switchWorkspace,
        toggleWorkspaceDropdown,
        toggleUserDropdown,
        updateWorkspaceDisplay,
        openCreateWorkspaceModal,
        closeCreateWorkspaceModal,
        submitCreateWorkspace,
        openWorkspaceSettings,

        // Register callback for workspace switch
        onWorkspaceSwitch(callback) {
            _onWorkspaceSwitchCallback = callback;
        },

        // Getters for state
        getCurrentWorkspace: () => currentWorkspace,
        getWorkspaces: () => workspaces,
        getCurrentUser: () => currentUser
    };
})();

// Expose toggle functions globally for onclick handlers in HTML
window.toggleWorkspaceDropdown = Navbar.toggleWorkspaceDropdown;
window.toggleUserDropdown = Navbar.toggleUserDropdown;
window.openCreateWorkspaceModal = Navbar.openCreateWorkspaceModal;
window.closeCreateWorkspaceModal = Navbar.closeCreateWorkspaceModal;
window.submitCreateWorkspace = Navbar.submitCreateWorkspace;
