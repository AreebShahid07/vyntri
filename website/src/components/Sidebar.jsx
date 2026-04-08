import { NavLink, Link, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';

export default function Sidebar() {
    const { pathname } = useLocation();
    const [activeSection, setActiveSection] = useState('');
    const [isMobileOpen, setIsMobileOpen] = useState(false);

    const docLinks = [
        {
            to: '/getting-started', label: 'Getting Started', sections: [
                { id: 'installation', label: 'Installation' },
                { id: 'requirements', label: 'Requirements' },
                { id: 'prepare-dataset', label: 'Prepare Dataset' },
                { id: 'first-model', label: 'Train First Model' },
                { id: 'predict', label: 'Make Predictions' },
                { id: 'continual', label: 'Continual Learning' }
            ]
        },
        {
            to: '/api', label: 'API Reference', sections: [
                { id: 'pipeline', label: 'Pipeline' },
                { id: 'config', label: 'Config' },
                { id: 'io', label: 'IO' },
                { id: 'engine', label: 'Dataset Engine' },
                { id: 'pvi', label: 'PVI' },
                { id: 'backbones', label: 'Backbones' },
                { id: 'solvers', label: 'Solvers' }
            ]
        },
        {
            to: '/solvers', label: 'Solvers Guide', sections: [
                { id: 'anacp', label: 'AnaCP' },
                { id: 'continual', label: 'ContinualAnaCP' },
                { id: 'fly', label: 'FlyCL' },
                { id: 'wisard', label: 'WiSARD' }
            ]
        },
        {
            to: '/cli', label: 'CLI Reference', sections: [
                { id: 'analyze', label: 'analyze' },
                { id: 'train', label: 'train' },
                { id: 'predict', label: 'predict' },
                { id: 'update', label: 'update' }
            ]
        }
    ];

    // Gather section IDs for the current page
    const currentPage = docLinks.find(l => l.to === pathname);
    const currentSectionIds = currentPage ? currentPage.sections.map(s => s.id) : [];

    useEffect(() => {
        setActiveSection('');
        if (currentSectionIds.length === 0) return;

        const observer = new IntersectionObserver(
            (entries) => {
                // Find the topmost visible section
                const visible = entries
                    .filter(e => e.isIntersecting)
                    .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
                if (visible.length > 0) {
                    setActiveSection(visible[0].target.id);
                }
            },
            { rootMargin: '-80px 0px -60% 0px', threshold: 0 }
        );

        const elements = currentSectionIds
            .map(id => document.getElementById(id))
            .filter(Boolean);
        elements.forEach(el => observer.observe(el));

        return () => observer.disconnect();
    }, [pathname]);

    useEffect(() => {
        setIsMobileOpen(false);
    }, [pathname, activeSection]);

    return (
        <div className="doc-sidebar">
            <div className="sidebar-content">
                <h3 className="sidebar-title">Documentation</h3>
                <button
                    type="button"
                    className="sidebar-toggle"
                    aria-expanded={isMobileOpen}
                    aria-controls="docs-sidebar-nav"
                    onClick={() => setIsMobileOpen((open) => !open)}
                >
                    <span>Documentation Navigation</span>
                    <span className={`sidebar-toggle-icon ${isMobileOpen ? 'open' : ''}`}>▾</span>
                </button>
                <nav id="docs-sidebar-nav" className={`sidebar-nav ${isMobileOpen ? 'open' : ''}`}>
                    {docLinks.map((link) => (
                        <div key={link.to} className={`sidebar-group ${pathname === link.to ? 'active-group' : ''}`}>
                            <NavLink
                                to={link.to}
                                className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}
                                end
                            >
                                {link.label}
                            </NavLink>
                            <ul className="sidebar-sections">
                                {link.sections.map(section => (
                                    <li key={section.id}>
                                        <Link
                                            to={`${link.to}#${section.id}`}
                                            className={`section-link${activeSection === section.id && pathname === link.to ? ' active' : ''}`}
                                        >
                                            {section.label}
                                        </Link>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </nav>
            </div>
        </div>
    );
}
