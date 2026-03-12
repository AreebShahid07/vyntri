import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import { useEffect } from 'react';

export default function DocLayout() {
    const { pathname, hash } = useLocation();

    useEffect(() => {
        if (hash) {
            setTimeout(() => {
                const id = hash.replace('#', '');
                const element = document.getElementById(id);
                if (element) {
                    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 100);
        } else {
            window.scrollTo(0, 0);
        }
    }, [pathname, hash]);

    return (
        <div className="doc-layout-container">
            <Sidebar />
            <div className="doc-layout-content">
                <Outlet />
            </div>
        </div>
    );
}
