import { NavLink, Link } from 'react-router-dom';

export default function Sidebar() {
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

    return (
        <div className="doc-sidebar">
            <div className="sidebar-content">
                <h3 className="sidebar-title">Documentation</h3>
                <nav className="sidebar-nav">
                    {docLinks.map((link) => (
                        <div key={link.to} className="sidebar-group">
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
                                        <Link to={`${link.to}#${section.id}`} className="section-link">
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
