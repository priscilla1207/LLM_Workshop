// ============================================
// CURRICULUM-NATIVE LLM — FRONTEND APP
// ============================================

const API_BASE = window.location.origin;

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    loadTopics();
    createParticles();

    // Enter key to submit
    document.getElementById('questionInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            askQuestion();
        }
    });
});

// ============================================
// ASK QUESTION — Main Flow
// ============================================
async function askQuestion() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    if (!question) return;

    const btn = document.getElementById('askBtn');
    btn.classList.add('loading');
    btn.disabled = true;

    // Show loading state
    showState('loading');
    animateLoadingSteps();

    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        displayAnswer(data);
    } catch (error) {
        displayError(error.message);
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

// ============================================
// DISPLAY ANSWER
// ============================================
function displayAnswer(data) {
    showState('answer');

    // Domain badge
    const badge = document.getElementById('domainBadge');
    const domainText = document.getElementById('domainText');
    if (data.is_in_curriculum) {
        badge.className = 'domain-badge in-domain';
        domainText.textContent = '✓ In Curriculum';
    } else {
        badge.className = 'domain-badge out-domain';
        domainText.textContent = '✗ Out of Scope';
    }

    // Question
    document.getElementById('answerQuestion').textContent = `"${data.question}"`;

    // Answer — render markdown-like formatting
    document.getElementById('answerBody').innerHTML = formatAnswer(data.answer);

    // Retrieved contexts
    const contextSection = document.getElementById('contextSection');
    const contextBody = document.getElementById('contextBody');

    if (data.retrieved_contexts && data.retrieved_contexts.length > 0) {
        contextSection.style.display = 'block';
        contextBody.innerHTML = data.retrieved_contexts.map(ctx =>
            `<div class="context-chunk">${escapeHtml(ctx)}</div>`
        ).join('');
    } else {
        contextSection.style.display = 'none';
    }
}

function displayError(message) {
    showState('answer');

    const badge = document.getElementById('domainBadge');
    badge.className = 'domain-badge out-domain';
    document.getElementById('domainText').textContent = 'Error';
    document.getElementById('answerQuestion').textContent = '';
    document.getElementById('answerBody').innerHTML = `
        <p style="color: var(--danger);">⚠ Failed to get answer: ${escapeHtml(message)}</p>
        <p style="color: var(--text-muted); font-size: 0.85rem;">Make sure the API server is running.</p>
    `;
    document.getElementById('contextSection').style.display = 'none';
}

// ============================================
// FORMAT ANSWER (simple markdown)
// ============================================
function formatAnswer(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Bold: **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Inline code: `code`
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');

    // Numbered lists: lines starting with 1. 2. etc
    html = html.replace(/^(\d+)\.\s+(.*)$/gm, '<li>$2</li>');

    // Bullet lists: lines starting with - or *
    html = html.replace(/^[-*]\s+(.*)$/gm, '<li>$1</li>');

    // Wrap consecutive <li> in <ul> or <ol>
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Paragraphs: split by double newlines
    html = html.split(/\n\n+/).map(block => {
        block = block.trim();
        if (!block) return '';
        if (block.startsWith('<ul>') || block.startsWith('<ol>')) return block;
        return `<p>${block.replace(/\n/g, '<br>')}</p>`;
    }).join('');

    return html;
}

// ============================================
// LOAD TOPICS
// ============================================
async function loadTopics() {
    try {
        const response = await fetch(`${API_BASE}/topics`);
        const data = await response.json();

        // Populate quick chips
        const chipsContainer = document.getElementById('quickChips');
        if (data.sample_questions) {
            chipsContainer.innerHTML = data.sample_questions.map(q =>
                `<button class="quick-chip" onclick="fillQuestion('${escapeAttr(q)}')">${escapeHtml(q)}</button>`
            ).join('');
        }

        // Populate topics list
        const topicsList = document.getElementById('topicsList');
        if (data.available_topics) {
            topicsList.innerHTML = data.available_topics.map(topic => {
                const isChapter = topic.startsWith('Chapter');
                const cls = isChapter ? 'chapter' : 'section';
                const dotCls = isChapter ? '' : 'section-dot';
                return `<div class="topic-item ${cls}">
                    <span class="topic-dot ${dotCls}"></span>
                    ${escapeHtml(topic)}
                </div>`;
            }).join('');
        }
    } catch (e) {
        console.warn('Could not load topics:', e);
    }
}

// ============================================
// UI HELPERS
// ============================================
function fillQuestion(q) {
    document.getElementById('questionInput').value = q;
    document.getElementById('questionInput').focus();
}

function showState(state) {
    document.getElementById('emptyState').style.display = state === 'empty' ? 'flex' : 'none';
    document.getElementById('loadingState').style.display = state === 'loading' ? 'flex' : 'none';
    document.getElementById('answerState').style.display = state === 'answer' ? 'block' : 'none';
}

function toggleTopics() {
    const list = document.getElementById('topicsList');
    const icon = document.getElementById('topicsExpandIcon');
    list.classList.toggle('expanded');
    icon.classList.toggle('expanded');
}

function toggleContext() {
    const body = document.getElementById('contextBody');
    const icon = document.getElementById('contextExpandIcon');
    body.classList.toggle('expanded');
    icon.classList.toggle('expanded');
}

function animateLoadingSteps() {
    const steps = [
        'Retrieving curriculum context...',
        'Matching syllabus topics...',
        'Constructing prompt...',
        'Groq LLM reasoning...',
        'Generating answer...'
    ];
    let i = 0;
    const el = document.getElementById('loadingStep');
    const interval = setInterval(() => {
        if (document.getElementById('loadingState').style.display === 'none') {
            clearInterval(interval);
            return;
        }
        i = (i + 1) % steps.length;
        el.textContent = steps[i];
    }, 1200);
}

// ============================================
// BACKGROUND PARTICLES
// ============================================
function createParticles() {
    const container = document.getElementById('particles');
    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDuration = (8 + Math.random() * 15) + 's';
        particle.style.animationDelay = Math.random() * 10 + 's';
        particle.style.width = (1 + Math.random() * 2) + 'px';
        particle.style.height = particle.style.width;
        container.appendChild(particle);
    }
}

// ============================================
// SANITIZATION HELPERS
// ============================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeAttr(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}
