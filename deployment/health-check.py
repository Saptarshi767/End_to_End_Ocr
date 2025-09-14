#!/usr/bin/env python3
"""
Health check script for OCR Analytics application
"""

import sys
import requests
import json
import time
from typing import Dict, Any, List
import psutil
import redis
import sqlalchemy
from datetime import datetime


class HealthChecker:
    """Comprehensive health check system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.base_url = self.config.get('base_url', 'http://localhost:8000')
        self.timeout = self.config.get('timeout', 30)
        self.checks = []
        
    def add_check(self, name: str, check_func, critical: bool = True):
        """Add a health check"""
        self.checks.append({
            'name': name,
            'func': check_func,
            'critical': critical
        })
    
    def check_api_health(self) -> Dict[str, Any]:
        """Check API health endpoint"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response_time': response.elapsed.total_seconds(),
                    'details': response.json() if response.content else {}
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': f"HTTP {response.status_code}",
                    'details': response.text
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_database_connection(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            database_url = self.config.get('database_url', 'sqlite:///./data/ocr_analytics.db')
            engine = sqlalchemy.create_engine(database_url)
            
            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text("SELECT 1"))
                result.fetchone()
            
            return {
                'status': 'healthy',
                'details': {'database_url': database_url.split('@')[-1] if '@' in database_url else database_url}
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_redis_connection(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379/0')
            r = redis.from_url(redis_url)
            
            # Test connection
            r.ping()
            
            # Get basic info
            info = r.info()
            
            return {
                'status': 'healthy',
                'details': {
                    'redis_version': info.get('redis_version'),
                    'connected_clients': info.get('connected_clients'),
                    'used_memory_human': info.get('used_memory_human')
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space"""
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            status = 'healthy'
            if free_percent < 10:
                status = 'critical'
            elif free_percent < 20:
                status = 'warning'
            
            return {
                'status': status,
                'details': {
                    'total_gb': round(disk_usage.total / (1024**3), 2),
                    'free_gb': round(disk_usage.free / (1024**3), 2),
                    'used_gb': round(disk_usage.used / (1024**3), 2),
                    'free_percent': round(free_percent, 2)
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            status = 'healthy'
            if memory.percent > 90:
                status = 'critical'
            elif memory.percent > 80:
                status = 'warning'
            
            return {
                'status': status,
                'details': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'used_percent': memory.percent
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            # Get CPU usage over 1 second
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = 'healthy'
            if cpu_percent > 90:
                status = 'critical'
            elif cpu_percent > 80:
                status = 'warning'
            
            return {
                'status': status,
                'details': {
                    'cpu_percent': cpu_percent,
                    'cpu_count': psutil.cpu_count()
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_ocr_engines(self) -> Dict[str, Any]:
        """Check OCR engines availability"""
        try:
            # Test Tesseract
            import subprocess
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            tesseract_available = result.returncode == 0
            tesseract_version = result.stdout.split('\n')[0] if tesseract_available else None
            
            # Test EasyOCR (optional)
            easyocr_available = False
            try:
                import easyocr
                easyocr_available = True
            except ImportError:
                pass
            
            engines = {
                'tesseract': {
                    'available': tesseract_available,
                    'version': tesseract_version
                },
                'easyocr': {
                    'available': easyocr_available
                }
            }
            
            # At least one engine should be available
            any_available = any(engine['available'] for engine in engines.values())
            
            return {
                'status': 'healthy' if any_available else 'unhealthy',
                'details': engines
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check file system permissions"""
        try:
            import os
            import tempfile
            
            directories = [
                self.config.get('data_dir', './data'),
                self.config.get('upload_dir', './uploads'),
                self.config.get('export_dir', './exports'),
                self.config.get('log_dir', './logs')
            ]
            
            results = {}
            all_writable = True
            
            for directory in directories:
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(directory, exist_ok=True)
                    
                    # Test write permission
                    test_file = os.path.join(directory, '.health_check_test')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    
                    results[directory] = {'writable': True}
                    
                except Exception as e:
                    results[directory] = {'writable': False, 'error': str(e)}
                    all_writable = False
            
            return {
                'status': 'healthy' if all_writable else 'unhealthy',
                'details': results
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        
        # Default checks
        default_checks = [
            ('api', self.check_api_health, True),
            ('database', self.check_database_connection, True),
            ('redis', self.check_redis_connection, False),
            ('disk_space', self.check_disk_space, True),
            ('memory', self.check_memory_usage, False),
            ('cpu', self.check_cpu_usage, False),
            ('ocr_engines', self.check_ocr_engines, True),
            ('file_permissions', self.check_file_permissions, True)
        ]
        
        # Add default checks if not already added
        existing_names = {check['name'] for check in self.checks}
        for name, func, critical in default_checks:
            if name not in existing_names:
                self.add_check(name, func, critical)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        critical_failures = []
        warnings = []
        
        for check in self.checks:
            try:
                result = check['func']()
                results['checks'][check['name']] = result
                
                if result['status'] == 'unhealthy':
                    if check['critical']:
                        critical_failures.append(check['name'])
                    else:
                        warnings.append(check['name'])
                elif result['status'] == 'critical':
                    critical_failures.append(check['name'])
                elif result['status'] == 'warning':
                    warnings.append(check['name'])
                    
            except Exception as e:
                results['checks'][check['name']] = {
                    'status': 'error',
                    'error': str(e)
                }
                if check['critical']:
                    critical_failures.append(check['name'])
        
        # Determine overall status
        if critical_failures:
            results['overall_status'] = 'unhealthy'
            results['critical_failures'] = critical_failures
        elif warnings:
            results['overall_status'] = 'warning'
            results['warnings'] = warnings
        
        return results


def main():
    """Main health check function"""
    
    # Configuration from environment or defaults
    config = {
        'base_url': 'http://localhost:8000',
        'timeout': 30,
        'database_url': 'sqlite:///./data/ocr_analytics.db',
        'redis_url': 'redis://localhost:6379/0',
        'data_dir': './data',
        'upload_dir': './uploads',
        'export_dir': './exports',
        'log_dir': './logs'
    }
    
    # Override with environment variables
    import os
    for key in config:
        env_key = key.upper()
        if env_key in os.environ:
            config[key] = os.environ[env_key]
    
    # Create health checker
    checker = HealthChecker(config)
    
    # Run checks
    results = checker.run_all_checks()
    
    # Output results
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        print(json.dumps(results, indent=2))
    else:
        # Human-readable output
        print(f"Health Check Results - {results['timestamp']}")
        print(f"Overall Status: {results['overall_status'].upper()}")
        print("-" * 50)
        
        for check_name, check_result in results['checks'].items():
            status = check_result['status'].upper()
            print(f"{check_name:20} {status}")
            
            if 'error' in check_result:
                print(f"{'':20} Error: {check_result['error']}")
            elif 'details' in check_result and isinstance(check_result['details'], dict):
                for key, value in check_result['details'].items():
                    print(f"{'':20} {key}: {value}")
        
        if 'critical_failures' in results:
            print(f"\nCritical Failures: {', '.join(results['critical_failures'])}")
        
        if 'warnings' in results:
            print(f"Warnings: {', '.join(results['warnings'])}")
    
    # Exit with appropriate code
    if results['overall_status'] == 'unhealthy':
        sys.exit(1)
    elif results['overall_status'] == 'warning':
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()