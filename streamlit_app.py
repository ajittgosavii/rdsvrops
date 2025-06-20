import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import requests
import json
import boto3
from datetime import datetime, timedelta
import base64
import io
import zipfile
from typing import Dict, List, Tuple, Any, Optional
import asyncio
import hashlib
import tempfile
import os
import anthropic 
import requests
import json
import urllib3
from datetime import datetime, timedelta

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# PDF Generation imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Add these enhanced recommendation display functions to your streamlit_app.py

def show_clear_recommendations_summary():
    """Show clear, actionable Reader/Writer recommendations"""
    
    st.markdown("## 🎯 Reader/Writer Recommendations Summary")
    
    if not hasattr(st.session_state, 'optimization_results') or not st.session_state.optimization_results:
        st.warning("⚠️ Please run the AI Optimizer to get Reader/Writer recommendations.")
        return
    
    optimization_results = st.session_state.optimization_results
    
    # Executive Summary Cards
    total_environments = len(optimization_results)
    total_writers = total_environments
    total_readers = sum([env['reader_optimization']['count'] for env in optimization_results.values()])
    total_monthly_cost = sum([env['cost_analysis']['monthly_breakdown']['total'] for env in optimization_results.values()])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Environments", total_environments)
    
    with col2:
        st.metric("✍️ Writer Instances", total_writers)
        
    with col3:
        st.metric("📖 Read Replicas", total_readers)
        
    with col4:
        st.metric("💰 Total Monthly Cost", f"${total_monthly_cost:,.0f}")
    
    # Detailed Recommendations Table
    st.markdown("### 📋 Detailed Recommendations")
    
    recommendations_data = []
    for env_name, optimization in optimization_results.items():
        writer = optimization['writer_optimization']
        reader = optimization['reader_optimization']
        
        recommendations_data.append({
            'Environment': env_name,
            'Current Status': '🔧 To Be Migrated',
            'Writer Instance': writer['instance_class'],
            'Writer Specs': f"{writer['specs'].vcpu} vCPU, {writer['specs'].memory_gb} GB RAM",
            'Reader Count': reader['count'],
            'Reader Instance': reader['instance_class'] if reader['count'] > 0 else 'None',
            'Multi-AZ': '✅ Yes' if writer['multi_az'] else '❌ No',
            'Monthly Cost': f"${optimization['cost_analysis']['monthly_breakdown']['total']:,.0f}",
            'Optimization Score': f"{optimization['optimization_score']:.0f}/100"
        })
    
    recommendations_df = pd.DataFrame(recommendations_data)
    st.dataframe(recommendations_df, use_container_width=True)
    
    # Action Items for Each Environment
    st.markdown("### 🚀 Implementation Action Items")
    
    for env_name, optimization in optimization_results.items():
        with st.expander(f"📋 {env_name} - Action Plan", expanded=False):
            writer = optimization['writer_optimization']
            reader = optimization['reader_optimization']
            
            st.markdown("#### ✅ Implementation Checklist")
            
            # Writer tasks
            st.markdown("**Writer Instance Setup:**")
            st.markdown(f"- [ ] Create {writer['instance_class']} RDS instance")
            st.markdown(f"- [ ] Configure {writer['specs'].vcpu} vCPUs and {writer['specs'].memory_gb} GB RAM")
            if writer['multi_az']:
                st.markdown("- [ ] Enable Multi-AZ deployment for high availability")
            st.markdown(f"- [ ] Estimated monthly cost: ${writer['monthly_cost']:,.0f}")
            
            # Reader tasks
            if reader['count'] > 0:
                st.markdown("**Read Replica Setup:**")
                st.markdown(f"- [ ] Create {reader['count']} read replica(s)")
                st.markdown(f"- [ ] Use {reader['instance_class']} instance type for each replica")
                st.markdown(f"- [ ] Estimated monthly cost for all replicas: ${reader['total_monthly_cost']:,.0f}")
                
                if reader['count'] > 1:
                    st.markdown("- [ ] Distribute replicas across different AZs for load balancing")
                st.markdown("- [ ] Configure application read traffic routing to replicas")
            else:
                st.markdown("**Read Replicas:** Not needed for this workload pattern")
            
            # Cost optimization tasks
            st.markdown("**Cost Optimization:**")
            ri_savings_1yr = optimization['cost_analysis']['reserved_instance_options']['1_year']['total_savings']
            ri_savings_3yr = optimization['cost_analysis']['reserved_instance_options']['3_year']['total_savings']
            
            if ri_savings_1yr > 1000:
                st.markdown(f"- [ ] Consider 1-year Reserved Instance (saves ${ri_savings_1yr:,.0f})")
            if ri_savings_3yr > 2000:
                st.markdown(f"- [ ] Consider 3-year Reserved Instance (saves ${ri_savings_3yr:,.0f})")
            
            # Monitoring tasks
            st.markdown("**Monitoring & Optimization:**")
            st.markdown("- [ ] Set up CloudWatch monitoring")
            st.markdown("- [ ] Configure performance alerts")
            st.markdown("- [ ] Review performance after 30 days")
            
            if reader['count'] > 0:
                st.markdown("- [ ] Monitor read replica lag")
                st.markdown("- [ ] Set up read traffic distribution monitoring")

def show_single_vs_bulk_comparison():
    """Show comparison between single instance and cluster recommendations"""
    
    st.markdown("### ⚖️ Single Instance vs Cluster Comparison")
    
    if not st.session_state.environment_specs:
        st.warning("Please configure environments first.")
        return
    
    # Create comparison for each environment
    for env_name, specs in st.session_state.environment_specs.items():
        with st.expander(f"🔍 {env_name} - Single vs Cluster Analysis"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🖥️ Single Instance Approach")
                
                # Calculate single instance recommendation
                total_cpu = specs.get('cpu_cores', 4)
                total_ram = specs.get('ram_gb', 16)
                
                if total_cpu <= 4 and total_ram <= 16:
                    single_instance = 'db.r5.large'
                    single_cost = 350
                elif total_cpu <= 8 and total_ram <= 32:
                    single_instance = 'db.r5.xlarge'
                    single_cost = 700
                elif total_cpu <= 16 and total_ram <= 64:
                    single_instance = 'db.r5.2xlarge'
                    single_cost = 1400
                else:
                    single_instance = 'db.r5.4xlarge'
                    single_cost = 2800
                
                st.info(f"""
                **Recommended Instance:** {single_instance}  
                **Configuration:** All-in-one  
                **Monthly Cost:** ${single_cost:,.0f}  
                **Pros:** Simple setup, lower cost  
                **Cons:** Single point of failure, limited read scaling
                """)
            
            with col2:
                st.markdown("#### 🔗 Writer/Reader Cluster Approach")
                
                # Get cluster recommendation if available
                if (hasattr(st.session_state, 'optimization_results') and 
                    st.session_state.optimization_results and 
                    env_name in st.session_state.optimization_results):
                    
                    opt = st.session_state.optimization_results[env_name]
                    writer = opt['writer_optimization']
                    reader = opt['reader_optimization']
                    total_cluster_cost = opt['cost_analysis']['monthly_breakdown']['total']
                    
                    st.success(f"""
                    **Writer:** {writer['instance_class']}  
                    **Readers:** {reader['count']} x {reader['instance_class'] if reader['count'] > 0 else 'None'}  
                    **Monthly Cost:** ${total_cluster_cost:,.0f}  
                    **Pros:** High availability, read scaling, better performance  
                    **Cons:** More complex, potentially higher cost
                    """)
                    
                    # Cost comparison
                    cost_diff = total_cluster_cost - single_cost
                    if cost_diff > 0:
                        st.warning(f"💰 Cluster costs ${cost_diff:,.0f} more per month (+{(cost_diff/single_cost)*100:.1f}%)")
                    else:
                        st.success(f"💰 Cluster saves ${abs(cost_diff):,.0f} per month ({abs(cost_diff/single_cost)*100:.1f}% savings)")
                else:
                    st.info("Run AI Optimizer to see cluster recommendations")
            
            # Recommendation
            st.markdown("#### 🎯 Our Recommendation")
            
            workload_pattern = specs.get('workload_pattern', 'balanced')
            env_type = specs.get('environment_type', 'production')
            
            if env_type == 'production' and workload_pattern in ['read_heavy', 'analytics']:
                st.success("✅ **Recommend Cluster Approach** - Production workload with read-heavy pattern benefits from read replicas")
            elif env_type == 'development':
                st.info("💡 **Recommend Single Instance** - Development environment prioritizes cost savings")
            else:
                st.warning("⚖️ **Evaluate Both Options** - Weigh cost vs performance benefits for your specific use case")

def show_performance_impact_analysis():
    """Show how recommendations will impact performance"""
    
    st.markdown("### 📈 Performance Impact Analysis")
    
    if not hasattr(st.session_state, 'optimization_results') or not st.session_state.optimization_results:
        st.info("Run AI Optimizer to see performance impact analysis.")
        return
    
    for env_name, optimization in st.session_state.optimization_results.items():
        with st.expander(f"📊 {env_name} - Performance Analysis"):
            
            workload = optimization['workload_analysis']
            writer = optimization['writer_optimization']
            reader = optimization['reader_optimization']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🔍 Workload Analysis")
                st.metric("Complexity Score", f"{workload['complexity_score']:.0f}/100")
                st.metric("CPU Intensity", f"{workload['cpu_intensity']:.0f}%")
                st.metric("Memory Intensity", f"{workload['memory_intensity']:.0f}%")
                st.metric("I/O Intensity", f"{workload['io_intensity']:.0f}%")
            
            with col2:
                st.markdown("#### ⚡ Performance Gains")
                
                # Calculate performance improvements
                current_cpu = st.session_state.environment_specs[env_name].get('cpu_cores', 4)
                recommended_cpu = writer['specs'].vcpu
                cpu_gain = (recommended_cpu / current_cpu - 1) * 100
                
                if cpu_gain > 0:
                    st.success(f"🚀 +{cpu_gain:.0f}% CPU performance")
                else:
                    st.info(f"💡 {abs(cpu_gain):.0f}% CPU optimization")
                
                current_ram = st.session_state.environment_specs[env_name].get('ram_gb', 16)
                recommended_ram = writer['specs'].memory_gb
                ram_gain = (recommended_ram / current_ram - 1) * 100
                
                if ram_gain > 0:
                    st.success(f"🧠 +{ram_gain:.0f}% memory capacity")
                else:
                    st.info(f"💡 {abs(ram_gain):.0f}% memory optimization")
                
                if reader['count'] > 0:
                    st.success(f"📖 {reader['count']}x read scaling capacity")
                
                if writer['multi_az']:
                    st.success("🛡️ 99.95% uptime SLA")
            
            with col3:
                st.markdown("#### 📋 Performance Recommendations")
                
                if workload['complexity_score'] > 70:
                    st.warning("⚠️ High complexity workload - monitor performance closely")
                else:
                    st.success("✅ Workload complexity manageable")
                
                if reader['count'] > 0:
                    st.info(f"💡 Route {workload['read_scaling_factor']:.1f}x read traffic to replicas")
                
                if workload['workload_type'] == 'read_heavy':
                    st.success("📈 Read replicas will significantly improve performance")
                elif workload['workload_type'] == 'write_heavy':
                    st.info("✍️ Focus on writer instance optimization")
                
                st.markdown("**Next Steps:**")
                st.markdown("- Set up performance monitoring")
                st.markdown("- Establish performance baselines")
                st.markdown("- Plan performance testing")

def show_migration_strategy_recommendations():
    """Show step-by-step migration strategy for Reader/Writer setup"""
    
    st.markdown("### 🗺️ Migration Strategy for Reader/Writer Architecture")
    
    if not hasattr(st.session_state, 'optimization_results') or not st.session_state.optimization_results:
        st.info("Run AI Optimizer to see migration strategy recommendations.")
        return
    
    # Migration phases
    phases = [
        {
            "phase": "Phase 1: Foundation Setup",
            "duration": "Week 1-2",
            "description": "Set up AWS infrastructure and prepare for migration",
            "tasks": [
                "Create VPC and subnets across multiple AZs",
                "Set up security groups and NACLs",
                "Configure parameter groups for target database engine",
                "Set up CloudWatch monitoring and alerting",
                "Create SNS topics for notifications"
            ]
        },
        {
            "phase": "Phase 2: Writer Instance Migration", 
            "duration": "Week 3-4",
            "description": "Migrate primary database to AWS as writer instance",
            "tasks": [
                "Create writer instance in primary AZ",
                "Use DMS for initial data migration",
                "Set up continuous replication",
                "Perform initial testing and validation",
                "Configure backup and maintenance windows"
            ]
        },
        {
            "phase": "Phase 3: Read Replica Deployment",
            "duration": "Week 5-6", 
            "description": "Add read replicas for scaling and availability",
            "tasks": [
                "Create read replicas in designated AZs",
                "Configure application connection strings for read traffic",
                "Implement read/write splitting logic",
                "Test read replica lag and performance",
                "Set up monitoring for replica health"
            ]
        },
        {
            "phase": "Phase 4: Application Migration",
            "duration": "Week 7-10",
            "description": "Migrate applications to use new database architecture", 
            "tasks": [
                "Update application configurations",
                "Test application functionality with new database",
                "Perform load testing with read/write splitting",
                "Validate data consistency across replicas",
                "Train team on new architecture"
            ]
        },
        {
            "phase": "Phase 5: Go-Live & Optimization",
            "duration": "Week 11-12",
            "description": "Complete cutover and optimize performance",
            "tasks": [
                "Execute final cutover during maintenance window",
                "Monitor performance and adjust as needed",
                "Optimize read traffic distribution",
                "Implement additional monitoring and alerting",
                "Document new architecture and procedures"
            ]
        }
    ]
    
    for i, phase in enumerate(phases, 1):
        with st.expander(f"📅 {phase['phase']} ({phase['duration']})", expanded=i<=2):
            st.markdown(f"**Duration:** {phase['duration']}")
            st.markdown(f"**Objective:** {phase['description']}")
            
            st.markdown("**Key Tasks:**")
            for task in phase['tasks']:
                st.markdown(f"- [ ] {task}")
            
            # Add environment-specific recommendations
            if i == 2:  # Writer setup phase
                st.markdown("**Environment-Specific Writer Setup:**")
                for env_name, optimization in st.session_state.optimization_results.items():
                    writer = optimization['writer_optimization']
                    st.markdown(f"• **{env_name}**: {writer['instance_class']} ({'Multi-AZ' if writer['multi_az'] else 'Single-AZ'})")
            
            elif i == 3:  # Reader setup phase
                st.markdown("**Environment-Specific Reader Setup:**")
                for env_name, optimization in st.session_state.optimization_results.items():
                    reader = optimization['reader_optimization']
                    if reader['count'] > 0:
                        st.markdown(f"• **{env_name}**: {reader['count']} x {reader['instance_class']}")
                    else:
                        st.markdown(f"• **{env_name}**: No read replicas needed")

# Add this to your main navigation or results dashboard
def show_enhanced_recommendations_dashboard():
    """Enhanced dashboard for Reader/Writer recommendations"""
    
    st.markdown("## 🎯 Complete Reader/Writer Recommendations")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Summary & Action Items",
        "⚖️ Single vs Cluster",
        "📈 Performance Impact", 
        "🗺️ Migration Strategy"
    ])
    
    with tab1:
        show_clear_recommendations_summary()
    
    with tab2:
        show_single_vs_bulk_comparison()
    
    with tab3:
        show_performance_impact_analysis()
    
    with tab4:
        show_migration_strategy_recommendations()

@dataclass
class InstanceSpecs:
    """Instance specifications with performance metrics"""
    instance_class: str
    vcpu: int
    memory_gb: float
    network_performance: str
    ebs_bandwidth_mbps: int
    hourly_cost: float
    suitable_for: List[str]
    max_connections: int
    iops_capability: int

class OptimizedReaderWriterAnalyzer:
    """Advanced Reader/Writer optimization with intelligent recommendations"""
    
    def __init__(self):
        self.instance_specs = self._initialize_instance_specs()
        self.pricing_data = self._initialize_pricing_data()
        
    def _initialize_instance_specs(self) -> Dict[str, InstanceSpecs]:
        """Initialize comprehensive instance specifications"""
        return {
            # T3 Series - Burstable Performance
            'db.t3.micro': InstanceSpecs('db.t3.micro', 2, 1, 'Low to Moderate', 87, 0.0255, ['development', 'testing'], 87, 1000),
            'db.t3.small': InstanceSpecs('db.t3.small', 2, 2, 'Low to Moderate', 174, 0.051, ['development', 'testing'], 174, 2000),
            'db.t3.medium': InstanceSpecs('db.t3.medium', 2, 4, 'Low to Moderate', 347, 0.102, ['development', 'small_production'], 347, 3000),
            'db.t3.large': InstanceSpecs('db.t3.large', 2, 8, 'Low to Moderate', 695, 0.204, ['development', 'small_production'], 695, 4000),
            'db.t3.xlarge': InstanceSpecs('db.t3.xlarge', 4, 16, 'Low to Moderate', 695, 0.408, ['staging', 'medium_production'], 1000, 5000),
            'db.t3.2xlarge': InstanceSpecs('db.t3.2xlarge', 8, 32, 'Low to Moderate', 695, 0.816, ['staging', 'medium_production'], 1500, 6000),
            
            # R5 Series - Memory Optimized
            'db.r5.large': InstanceSpecs('db.r5.large', 2, 16, 'Up to 10 Gbps', 693, 0.24, ['memory_intensive', 'analytics'], 1000, 7500),
            'db.r5.xlarge': InstanceSpecs('db.r5.xlarge', 4, 32, 'Up to 10 Gbps', 1387, 0.48, ['memory_intensive', 'production'], 2000, 10000),
            'db.r5.2xlarge': InstanceSpecs('db.r5.2xlarge', 8, 64, 'Up to 10 Gbps', 2775, 0.96, ['large_production', 'analytics'], 3000, 15000),
            'db.r5.4xlarge': InstanceSpecs('db.r5.4xlarge', 16, 128, '10 Gbps', 4750, 1.92, ['large_production', 'high_memory'], 5000, 25000),
            'db.r5.8xlarge': InstanceSpecs('db.r5.8xlarge', 32, 256, '10 Gbps', 6800, 3.84, ['enterprise', 'high_performance'], 8000, 40000),
            'db.r5.12xlarge': InstanceSpecs('db.r5.12xlarge', 48, 384, '12 Gbps', 9500, 5.76, ['enterprise', 'very_high_memory'], 12000, 60000),
            'db.r5.16xlarge': InstanceSpecs('db.r5.16xlarge', 64, 512, '20 Gbps', 13600, 7.68, ['enterprise', 'extreme_performance'], 16000, 80000),
            'db.r5.24xlarge': InstanceSpecs('db.r5.24xlarge', 96, 768, '25 Gbps', 19000, 11.52, ['enterprise', 'maximum_performance'], 24000, 120000),
            
            # R6i Series - Latest Generation Memory Optimized
            'db.r6i.large': InstanceSpecs('db.r6i.large', 2, 16, 'Up to 12.5 Gbps', 1000, 0.252, ['memory_intensive', 'latest_gen'], 1200, 8000),
            'db.r6i.xlarge': InstanceSpecs('db.r6i.xlarge', 4, 32, 'Up to 12.5 Gbps', 2000, 0.504, ['production', 'latest_gen'], 2400, 12000),
            'db.r6i.2xlarge': InstanceSpecs('db.r6i.2xlarge', 8, 64, 'Up to 12.5 Gbps', 4000, 1.008, ['large_production', 'latest_gen'], 4000, 18000),
            'db.r6i.4xlarge': InstanceSpecs('db.r6i.4xlarge', 16, 128, '12.5 Gbps', 8000, 2.016, ['enterprise', 'latest_gen'], 6000, 30000),
            
            # C5 Series - Compute Optimized
            'db.c5.large': InstanceSpecs('db.c5.large', 2, 4, 'Up to 10 Gbps', 693, 0.192, ['compute_intensive', 'low_memory'], 1000, 5000),
            'db.c5.xlarge': InstanceSpecs('db.c5.xlarge', 4, 8, 'Up to 10 Gbps', 1387, 0.384, ['compute_intensive', 'oltp'], 2000, 8000),
            'db.c5.2xlarge': InstanceSpecs('db.c5.2xlarge', 8, 16, 'Up to 10 Gbps', 2775, 0.768, ['high_cpu', 'oltp'], 3000, 12000),
            'db.c5.4xlarge': InstanceSpecs('db.c5.4xlarge', 16, 32, '10 Gbps', 4750, 1.536, ['very_high_cpu', 'oltp'], 5000, 20000),
        }
    
    def _initialize_pricing_data(self) -> Dict:
        """Initialize comprehensive pricing data for different regions and deployment options"""
        return {
            'us-east-1': {
                'reserved_1_year': {'discount': 0.35, 'upfront_ratio': 0.0},
                'reserved_3_year': {'discount': 0.55, 'upfront_ratio': 0.0},
                'spot': {'discount': 0.70, 'availability': 0.85},
                'multi_az_multiplier': 2.0,
                'cross_az_transfer_gb': 0.01,
                'backup_storage_gb': 0.095,
                'snapshot_storage_gb': 0.095
            }
        }
    
    def optimize_cluster_configuration(self, environment_specs: Dict) -> Dict:
        """Generate optimized Reader/Writer recommendations with detailed cost analysis"""
        
        optimized_recommendations = {}
        
        for env_name, specs in environment_specs.items():
            optimization = self._optimize_single_environment(env_name, specs)
            optimized_recommendations[env_name] = optimization
        
        return optimized_recommendations
    
    def _optimize_single_environment(self, env_name: str, specs: Dict) -> Dict:
        """Optimize configuration for a single environment"""
        
        # Extract environment characteristics
        cpu_cores = specs.get('cpu_cores', 4)
        ram_gb = specs.get('ram_gb', 16)
        storage_gb = specs.get('storage_gb', 500)
        iops_requirement = specs.get('iops_requirement', 3000)
        peak_connections = specs.get('peak_connections', 100)
        workload_pattern = specs.get('workload_pattern', 'balanced')
        read_write_ratio = specs.get('read_write_ratio', 70)
        environment_type = specs.get('environment_type', 'production')
        daily_usage_hours = specs.get('daily_usage_hours', 24)
        
        # Analyze workload characteristics
        workload_analysis = self._analyze_workload_characteristics(
            cpu_cores, ram_gb, iops_requirement, peak_connections, 
            workload_pattern, read_write_ratio
        )
        
        # Optimize Writer configuration
        writer_optimization = self._optimize_writer_instance(workload_analysis, environment_type)
        
        # Optimize Reader configuration
        reader_optimization = self._optimize_reader_configuration(
            writer_optimization, workload_analysis, environment_type
        )
        
        # Calculate comprehensive costs
        cost_analysis = self._calculate_comprehensive_costs(
            writer_optimization, reader_optimization, storage_gb, 
            iops_requirement, environment_type, daily_usage_hours
        )
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            workload_analysis, writer_optimization, reader_optimization, cost_analysis
        )
        
        return {
            'environment_name': env_name,
            'environment_type': environment_type,
            'workload_analysis': workload_analysis,
            'writer_optimization': writer_optimization,
            'reader_optimization': reader_optimization,
            'cost_analysis': cost_analysis,
            'recommendations': recommendations,
            'optimization_score': self._calculate_optimization_score(
                workload_analysis, writer_optimization, reader_optimization, cost_analysis
            )
        }
    
    def _analyze_workload_characteristics(self, cpu_cores: int, ram_gb: int, 
                                        iops_requirement: int, peak_connections: int,
                                        workload_pattern: str, read_write_ratio: int) -> Dict:
        """Analyze workload characteristics to determine optimal configuration"""
        
        # Calculate workload intensity
        cpu_intensity = min(100, (cpu_cores / 64) * 100)  # Normalize to 64 cores max
        memory_intensity = min(100, (ram_gb / 768) * 100)  # Normalize to 768 GB max
        io_intensity = min(100, (iops_requirement / 80000) * 100)  # Normalize to 80K IOPS max
        connection_intensity = min(100, (peak_connections / 16000) * 100)  # Normalize to 16K connections max
        
        # Determine workload type
        if workload_pattern == 'read_heavy' or read_write_ratio >= 80:
            workload_type = 'read_heavy'
            read_scaling_factor = 2.0
        elif workload_pattern == 'write_heavy' or read_write_ratio <= 30:
            workload_type = 'write_heavy'
            read_scaling_factor = 0.5
        elif workload_pattern == 'analytics':
            workload_type = 'analytics'
            read_scaling_factor = 3.0
        else:
            workload_type = 'balanced'
            read_scaling_factor = 1.2
        
        # Calculate complexity score
        complexity_score = (cpu_intensity + memory_intensity + io_intensity + connection_intensity) / 4
        
        return {
            'cpu_intensity': cpu_intensity,
            'memory_intensity': memory_intensity,
            'io_intensity': io_intensity,
            'connection_intensity': connection_intensity,
            'complexity_score': complexity_score,
            'workload_type': workload_type,
            'read_scaling_factor': read_scaling_factor,
            'recommended_reader_count': self._calculate_optimal_reader_count(
                workload_type, complexity_score, peak_connections
            ),
            'performance_requirements': self._determine_performance_requirements(
                cpu_intensity, memory_intensity, io_intensity
            )
        }
    
    def _calculate_optimal_reader_count(self, workload_type: str, complexity_score: float, peak_connections: int) -> int:
        """Calculate optimal number of read replicas"""
        
        base_readers = {
            'read_heavy': 3,
            'analytics': 2,
            'balanced': 1,
            'write_heavy': 0
        }
        
        # Adjust based on complexity and connections
        complexity_adjustment = int(complexity_score / 30)  # Add reader every 30 points of complexity
        connection_adjustment = int(peak_connections / 2000)  # Add reader every 2000 connections
        
        optimal_count = base_readers.get(workload_type, 1) + complexity_adjustment + connection_adjustment
        
        return min(5, max(0, optimal_count))  # Cap at 5 readers
    
    def _determine_performance_requirements(self, cpu_intensity: float, memory_intensity: float, io_intensity: float) -> str:
        """Determine performance tier requirements"""
        
        avg_intensity = (cpu_intensity + memory_intensity + io_intensity) / 3
        
        if avg_intensity >= 80:
            return 'high_performance'
        elif avg_intensity >= 60:
            return 'medium_performance'
        elif avg_intensity >= 40:
            return 'standard_performance'
        else:
            return 'basic_performance'
    
    def _optimize_writer_instance(self, workload_analysis: Dict, environment_type: str) -> Dict:
        """Optimize Writer instance selection"""
        
        performance_req = workload_analysis['performance_requirements']
        complexity_score = workload_analysis['complexity_score']
        
        # Filter suitable instances based on performance requirements
        suitable_instances = []
        
        for instance_class, specs in self.instance_specs.items():
            if self._is_instance_suitable_for_writer(specs, performance_req, environment_type, complexity_score):
                suitable_instances.append((instance_class, specs))
        
        # Score instances based on multiple criteria
        scored_instances = []
        for instance_class, specs in suitable_instances:
            score = self._score_writer_instance(specs, workload_analysis, environment_type)
            scored_instances.append((score, instance_class, specs))
        
        # Sort by score (higher is better)
        scored_instances.sort(reverse=True)
        
        # Get top recommendations
        primary_recommendation = scored_instances[0] if scored_instances else None
        alternatives = scored_instances[1:3] if len(scored_instances) > 1 else []
        
        if primary_recommendation:
            score, instance_class, specs = primary_recommendation
            
            return {
                'instance_class': instance_class,
                'specs': specs,
                'score': score,
                'multi_az': environment_type in ['production', 'staging'],
                'reasoning': self._generate_writer_reasoning(specs, workload_analysis, environment_type),
                'alternatives': [
                    {'instance_class': alt[1], 'score': alt[0], 'specs': alt[2]}
                    for alt in alternatives
                ],
                'monthly_cost': self._calculate_instance_monthly_cost(specs, environment_type),
                'annual_cost': self._calculate_instance_monthly_cost(specs, environment_type) * 12
            }
        else:
            # Fallback if no suitable instances found
            fallback_instance = 'db.r5.large'
            fallback_specs = self.instance_specs[fallback_instance]
            
            return {
                'instance_class': fallback_instance,
                'specs': fallback_specs,
                'score': 50.0,
                'multi_az': True,
                'reasoning': 'Fallback recommendation due to no optimal matches',
                'alternatives': [],
                'monthly_cost': self._calculate_instance_monthly_cost(fallback_specs, environment_type),
                'annual_cost': self._calculate_instance_monthly_cost(fallback_specs, environment_type) * 12
            }
    
    def _is_instance_suitable_for_writer(self, specs: InstanceSpecs, performance_req: str, 
                                       environment_type: str, complexity_score: float) -> bool:
        """Check if instance is suitable for writer role"""
        
        # Performance tier requirements
        performance_minimums = {
            'high_performance': {'min_vcpu': 16, 'min_memory': 64, 'min_iops': 20000},
            'medium_performance': {'min_vcpu': 8, 'min_memory': 32, 'min_iops': 10000},
            'standard_performance': {'min_vcpu': 4, 'min_memory': 16, 'min_iops': 5000},
            'basic_performance': {'min_vcpu': 2, 'min_memory': 8, 'min_iops': 2000}
        }
        
        minimums = performance_minimums.get(performance_req, performance_minimums['standard_performance'])
        
        # Check basic requirements
        if (specs.vcpu < minimums['min_vcpu'] or 
            specs.memory_gb < minimums['min_memory'] or 
            specs.iops_capability < minimums['min_iops']):
            return False
        
        # Environment-specific filters
        if environment_type == 'production':
            # Production requires more robust instances
            if specs.vcpu < 4 or specs.memory_gb < 16:
                return False
            # Avoid burstable instances for production unless complexity is low
            if 't3' in specs.instance_class and complexity_score > 50:
                return False
        
        return True
    
    def _score_writer_instance(self, specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> float:
        """Score writer instance based on multiple criteria"""
        
        score = 0.0
        
        # Performance scoring (40% weight)
        performance_score = min(100, (specs.vcpu * 10 + specs.memory_gb / 4 + specs.iops_capability / 500))
        score += performance_score * 0.4
        
        # Cost efficiency scoring (30% weight)
        cost_per_vcpu = specs.hourly_cost / specs.vcpu
        cost_per_gb_memory = specs.hourly_cost / specs.memory_gb
        cost_efficiency = max(0, 100 - (cost_per_vcpu * 100 + cost_per_gb_memory * 10))
        score += cost_efficiency * 0.3
        
        # Suitability scoring (20% weight)
        suitability_score = 0
        if environment_type in specs.suitable_for:
            suitability_score = 100
        elif any(env in specs.suitable_for for env in ['production', 'large_production', 'enterprise']):
            suitability_score = 80
        else:
            suitability_score = 60
        score += suitability_score * 0.2
        
        # Network performance scoring (10% weight)
        network_score = 60  # Base score
        if '25 Gbps' in specs.network_performance:
            network_score = 100
        elif '20 Gbps' in specs.network_performance:
            network_score = 90
        elif '12' in specs.network_performance:
            network_score = 80
        elif '10 Gbps' in specs.network_performance:
            network_score = 70
        score += network_score * 0.1
        
        return min(100, score)
    
    def _optimize_reader_configuration(self, writer_optimization: Dict, workload_analysis: Dict, environment_type: str) -> Dict:
        """Optimize Reader configuration based on writer and workload"""
        
        recommended_count = workload_analysis['recommended_reader_count']
        
        if recommended_count == 0:
            return {
                'count': 0,
                'instance_class': None,
                'specs': None,
                'total_monthly_cost': 0,
                'total_annual_cost': 0,
                'reasoning': 'No read replicas needed for this workload pattern'
            }
        
        # Determine optimal reader instance size
        writer_specs = writer_optimization['specs']
        reader_instance_class = self._calculate_optimal_reader_size(writer_specs, workload_analysis, environment_type)
        reader_specs = self.instance_specs[reader_instance_class]
        
        # Calculate costs
        single_reader_monthly_cost = self._calculate_instance_monthly_cost(reader_specs, environment_type, is_reader=True)
        total_monthly_cost = single_reader_monthly_cost * recommended_count
        total_annual_cost = total_monthly_cost * 12
        
        return {
            'count': recommended_count,
            'instance_class': reader_instance_class,
            'specs': reader_specs,
            'single_reader_monthly_cost': single_reader_monthly_cost,
            'total_monthly_cost': total_monthly_cost,
            'total_annual_cost': total_annual_cost,
            'multi_az': environment_type == 'production',
            'reasoning': self._generate_reader_reasoning(recommended_count, reader_specs, workload_analysis),
            'scaling_recommendations': self._generate_reader_scaling_recommendations(workload_analysis)
        }
    
    def _calculate_optimal_reader_size(self, writer_specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> str:
        """Calculate optimal reader instance size"""
        
        workload_type = workload_analysis['workload_type']
        read_scaling_factor = workload_analysis['read_scaling_factor']
        
        # Base strategy: readers are typically same size or smaller than writer
        if workload_type == 'read_heavy' or workload_type == 'analytics':
            # For read-heavy workloads, readers should match or exceed writer capacity
            target_vcpu = int(writer_specs.vcpu * read_scaling_factor)
            target_memory = writer_specs.memory_gb * read_scaling_factor
        else:
            # For balanced/write-heavy workloads, readers can be smaller
            target_vcpu = max(2, int(writer_specs.vcpu * 0.7))
            target_memory = writer_specs.memory_gb * 0.7
        
        # Find best matching instance
        best_match = None
        best_score = 0
        
        for instance_class, specs in self.instance_specs.items():
            if specs.vcpu >= target_vcpu * 0.8 and specs.memory_gb >= target_memory * 0.8:
                # Score based on how close it matches our target and cost efficiency
                size_match_score = 100 - abs(specs.vcpu - target_vcpu) * 5 - abs(specs.memory_gb - target_memory) * 2
                cost_efficiency_score = 100 - (specs.hourly_cost * 10)  # Lower cost is better
                overall_score = size_match_score * 0.7 + cost_efficiency_score * 0.3
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_match = instance_class
        
        return best_match or 'db.r5.large'  # Fallback
    
    def _calculate_instance_monthly_cost(self, specs: InstanceSpecs, environment_type: str, is_reader: bool = False) -> float:
        """Calculate monthly cost for an instance"""
        
        base_hourly_cost = specs.hourly_cost
        
        # Apply Multi-AZ multiplier if needed
        if environment_type in ['production', 'staging'] and not is_reader:
            base_hourly_cost *= self.pricing_data['us-east-1']['multi_az_multiplier']
        
        # Calculate monthly cost (730 hours per month)
        monthly_cost = base_hourly_cost * 730
        
        return monthly_cost
    
    def _calculate_comprehensive_costs(self, writer_optimization: Dict, reader_optimization: Dict,
                                     storage_gb: int, iops_requirement: int, environment_type: str,
                                     daily_usage_hours: int) -> Dict:
        """Calculate comprehensive cost analysis"""
        
        # Instance costs
        writer_monthly_cost = writer_optimization['monthly_cost']
        reader_monthly_cost = reader_optimization['total_monthly_cost']
        total_instance_monthly_cost = writer_monthly_cost + reader_monthly_cost
        
        # Storage costs
        storage_costs = self._calculate_storage_costs(storage_gb, iops_requirement, environment_type)
        
        # Additional costs
        backup_monthly_cost = storage_gb * 0.095  # $0.095 per GB per month
        monitoring_monthly_cost = 50 if environment_type == 'production' else 20  # CloudWatch Enhanced Monitoring
        
        # Cross-AZ transfer costs (estimated)
        cross_az_monthly_cost = 0
        if reader_optimization['count'] > 0:
            estimated_cross_az_gb = storage_gb * 0.1  # Estimate 10% of storage as cross-AZ traffic
            cross_az_monthly_cost = estimated_cross_az_gb * 0.01
        
        total_monthly_cost = (total_instance_monthly_cost + storage_costs['total_monthly'] + 
                            backup_monthly_cost + monitoring_monthly_cost + cross_az_monthly_cost)
        
        # Reserved Instance calculations
        reserved_1_year = self._calculate_reserved_instance_savings(total_instance_monthly_cost, 1)
        reserved_3_year = self._calculate_reserved_instance_savings(total_instance_monthly_cost, 3)
        
        return {
            'monthly_breakdown': {
                'writer_instance': writer_monthly_cost,
                'reader_instances': reader_monthly_cost,
                'storage': storage_costs['total_monthly'],
                'backup': backup_monthly_cost,
                'monitoring': monitoring_monthly_cost,
                'cross_az_transfer': cross_az_monthly_cost,
                'total': total_monthly_cost
            },
            'annual_breakdown': {
                'writer_instance': writer_monthly_cost * 12,
                'reader_instances': reader_monthly_cost * 12,
                'storage': storage_costs['total_monthly'] * 12,
                'backup': backup_monthly_cost * 12,
                'monitoring': monitoring_monthly_cost * 12,
                'cross_az_transfer': cross_az_monthly_cost * 12,
                'total': total_monthly_cost * 12
            },
            'storage_details': storage_costs,
            'reserved_instance_options': {
                '1_year': reserved_1_year,
                '3_year': reserved_3_year
            },
            'cost_optimization_opportunities': self._identify_cost_optimization_opportunities(
                writer_optimization, reader_optimization, storage_costs, environment_type
            )
        }
    
    def _calculate_storage_costs(self, storage_gb: int, iops_requirement: int, environment_type: str) -> Dict:
        """Calculate detailed storage costs"""
        
        # Determine optimal storage type
        if iops_requirement > 16000:
            storage_type = 'io2'
            base_cost_per_gb = 0.125
            iops_cost_per_iops = 0.065
        elif iops_requirement > 3000:
            storage_type = 'gp3'
            base_cost_per_gb = 0.08
            # GP3 includes 3000 IOPS free, charge for additional
            additional_iops = max(0, iops_requirement - 3000)
            iops_cost_per_iops = 0.005 if additional_iops > 0 else 0
        else:
            storage_type = 'gp3'
            base_cost_per_gb = 0.08
            iops_cost_per_iops = 0
            additional_iops = 0
        
        base_storage_cost = storage_gb * base_cost_per_gb
        iops_cost = (additional_iops if storage_type == 'gp3' else iops_requirement) * iops_cost_per_iops
        total_monthly_cost = base_storage_cost + iops_cost
        
        return {
            'storage_type': storage_type,
            'storage_gb': storage_gb,
            'iops_provisioned': iops_requirement,
            'base_storage_cost': base_storage_cost,
            'iops_cost': iops_cost,
            'total_monthly': total_monthly_cost,
            'cost_per_gb': base_cost_per_gb,
            'cost_per_iops': iops_cost_per_iops
        }
    
    def _calculate_reserved_instance_savings(self, monthly_instance_cost: float, years: int) -> Dict:
        """Calculate Reserved Instance savings"""
        
        if years == 1:
            discount = self.pricing_data['us-east-1']['reserved_1_year']['discount']
        else:
            discount = self.pricing_data['us-east-1']['reserved_3_year']['discount']
        
        annual_on_demand = monthly_instance_cost * 12
        annual_reserved = annual_on_demand * (1 - discount)
        total_on_demand = annual_on_demand * years
        total_reserved = annual_reserved * years
        total_savings = total_on_demand - total_reserved
        
        return {
            'term_years': years,
            'discount_percentage': discount * 100,
            'annual_cost': annual_reserved,
            'total_cost': total_reserved,
            'total_savings': total_savings,
            'monthly_cost': annual_reserved / 12
        }
    
    def _identify_cost_optimization_opportunities(self, writer_optimization: Dict, 
                                                reader_optimization: Dict, storage_costs: Dict,
                                                environment_type: str) -> List[Dict]:
        """Identify cost optimization opportunities"""
        
        opportunities = []
        
        # Reserved Instance opportunity
        if writer_optimization['monthly_cost'] > 200:  # Threshold for RI consideration
            opportunities.append({
                'type': 'Reserved Instances',
                'description': 'Consider 1-year or 3-year Reserved Instances for predictable workloads',
                'potential_savings': writer_optimization['monthly_cost'] * 12 * 0.35,
                'impact': 'High',
                'implementation_effort': 'Low'
            })
        
        # Reader optimization
        if reader_optimization['count'] > 2:
            opportunities.append({
                'type': 'Reader Instance Optimization',
                'description': 'Consider using Aurora Auto Scaling for variable read workloads',
                'potential_savings': reader_optimization['total_monthly_cost'] * 0.2,
                'impact': 'Medium',
                'implementation_effort': 'Medium'
            })
        
        # Storage optimization
        if storage_costs['storage_type'] == 'io2' and storage_costs['iops_provisioned'] < 10000:
            opportunities.append({
                'type': 'Storage Type Optimization',
                'description': 'Consider gp3 storage for lower IOPS requirements',
                'potential_savings': storage_costs['total_monthly'] * 0.3,
                'impact': 'Medium',
                'implementation_effort': 'Low'
            })
        
        # Environment-specific optimizations
        if environment_type in ['development', 'testing']:
            opportunities.append({
                'type': 'Development Environment Optimization',
                'description': 'Consider scheduled start/stop or spot instances for non-production',
                'potential_savings': (writer_optimization['monthly_cost'] + reader_optimization['total_monthly_cost']) * 0.5,
                'impact': 'High',
                'implementation_effort': 'Medium'
            })
        
        return opportunities
    
    def _generate_writer_reasoning(self, specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> str:
        """Generate reasoning for writer instance recommendation"""
        
        reasoning_parts = []
        
        # Performance reasoning
        if workload_analysis['complexity_score'] > 70:
            reasoning_parts.append(f"High complexity workload requires {specs.vcpu} vCPUs and {specs.memory_gb}GB memory")
        else:
            reasoning_parts.append(f"Balanced configuration with {specs.vcpu} vCPUs and {specs.memory_gb}GB memory")
        
        # Environment reasoning
        if environment_type == 'production':
            reasoning_parts.append("Production-grade instance with high availability features")
        else:
            reasoning_parts.append(f"Cost-optimized for {environment_type} environment")
        
        # Network reasoning
        if '25 Gbps' in specs.network_performance:
            reasoning_parts.append("Enhanced networking for high-throughput workloads")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_reader_reasoning(self, count: int, specs: InstanceSpecs, workload_analysis: Dict) -> str:
        """Generate reasoning for reader configuration"""
        
        if count == 0:
            return "No read replicas needed for write-heavy or low-complexity workloads"
        
        workload_type = workload_analysis['workload_type']
        
        reasoning_parts = [
            f"{count} read replica{'s' if count > 1 else ''} recommended for {workload_type} workload",
            f"Each reader: {specs.vcpu} vCPUs, {specs.memory_gb}GB memory"
        ]
        
        if workload_type == 'read_heavy':
            reasoning_parts.append("Multiple readers will distribute read load effectively")
        elif workload_type == 'analytics':
            reasoning_parts.append("Dedicated readers for analytical queries to avoid impact on primary")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_reader_scaling_recommendations(self, workload_analysis: Dict) -> List[str]:
        """Generate reader scaling recommendations"""
        
        recommendations = []
        
        if workload_analysis['workload_type'] == 'read_heavy':
            recommendations.append("Consider Aurora Auto Scaling to automatically adjust reader count based on CPU/connection metrics")
            recommendations.append("Monitor read replica lag and add readers if lag consistently exceeds 1 second")
        
        if workload_analysis['complexity_score'] > 60:
            recommendations.append("For high-complexity workloads, consider reader instances same size as writer")
        
        recommendations.append("Use connection pooling (like PgBouncer) to optimize connection distribution across readers")
        
        return recommendations
    
    def _generate_optimization_recommendations(self, workload_analysis: Dict, writer_optimization: Dict,
                                             reader_optimization: Dict, cost_analysis: Dict) -> List[str]:
        """Generate comprehensive optimization recommendations"""
        
        recommendations = []
        
        # Performance recommendations
        if workload_analysis['complexity_score'] > 80:
            recommendations.append("Consider upgrading to latest generation instances (R6i) for better price/performance")
        
        # Cost recommendations
        total_annual_cost = cost_analysis['annual_breakdown']['total']
        if total_annual_cost > 50000:
            recommendations.append("Evaluate Reserved Instances for significant cost savings on predictable workloads")
        
        # Scaling recommendations
        if reader_optimization['count'] > 1:
            recommendations.append("Implement Aurora Auto Scaling to optimize reader count based on actual demand")
        
        # Monitoring recommendations
        recommendations.append("Set up Enhanced Monitoring and Performance Insights for optimization opportunities")
        recommendations.append("Configure CloudWatch alarms for CPU, memory, and connection metrics")
        
        return recommendations
    
    def _calculate_optimization_score(self, workload_analysis: Dict, writer_optimization: Dict,
                                    reader_optimization: Dict, cost_analysis: Dict) -> float:
        """Calculate overall optimization score"""
        
        # Performance score (0-100)
        performance_score = min(100, writer_optimization['score'])
        
        # Cost efficiency score (0-100)
        cost_per_vcpu = cost_analysis['monthly_breakdown']['total'] / writer_optimization['specs'].vcpu
        cost_efficiency_score = max(0, 100 - (cost_per_vcpu / 10))  # Lower cost per vCPU = higher score
        
        # Configuration appropriateness (0-100)
        config_score = 80  # Base score
        if reader_optimization['count'] > 0 and workload_analysis['workload_type'] in ['read_heavy', 'analytics']:
            config_score += 15
        elif reader_optimization['count'] == 0 and workload_analysis['workload_type'] == 'write_heavy':
            config_score += 15
        
        # Overall score (weighted average)
        overall_score = (performance_score * 0.4 + cost_efficiency_score * 0.4 + config_score * 0.2)
        
        return min(100, overall_score)

def display_optimization_results(optimization_results: Dict):
    """Display comprehensive optimization results in Streamlit"""
    
    st.markdown("# 🚀 Optimized Reader/Writer Recommendations")
    
    # Overall summary
    total_monthly_cost = sum([env['cost_analysis']['monthly_breakdown']['total'] 
                             for env in optimization_results.values()])
    total_annual_cost = total_monthly_cost * 12
    avg_optimization_score = sum([env['optimization_score'] 
                                 for env in optimization_results.values()]) / len(optimization_results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Monthly Cost", f"${total_monthly_cost:,.0f}")
    
    with col2:
        st.metric("Total Annual Cost", f"${total_annual_cost:,.0f}")
    
    with col3:
        st.metric("Avg Optimization Score", f"{avg_optimization_score:.1f}/100")
    
    with col4:
        total_instances = sum([1 + env['reader_optimization']['count'] 
                              for env in optimization_results.values()])
        st.metric("Total Instances", total_instances)
    
    # Detailed results for each environment
    for env_name, optimization in optimization_results.items():
        with st.expander(f"🏢 {env_name.title()} - Optimization Score: {optimization['optimization_score']:.1f}/100", expanded=True):
            
            # Configuration overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✍️ Writer Configuration")
                writer = optimization['writer_optimization']
                st.info(f"""
                **Instance:** {writer['instance_class']}  
                **vCPUs:** {writer['specs'].vcpu}  
                **Memory:** {writer['specs'].memory_gb} GB  
                **Network:** {writer['specs'].network_performance}  
                **Multi-AZ:** {'✅ Yes' if writer['multi_az'] else '❌ No'}  
                **Monthly Cost:** ${writer['monthly_cost']:,.0f}  
                **Annual Cost:** ${writer['annual_cost']:,.0f}
                """)
                
                st.markdown("**Reasoning:**")
                st.write(writer['reasoning'])
            
            with col2:
                st.markdown("#### 📖 Reader Configuration")
                reader = optimization['reader_optimization']
                
                if reader['count'] > 0:
                    st.success(f"""
                    **Count:** {reader['count']} replicas  
                    **Instance:** {reader['instance_class']}  
                    **vCPUs:** {reader['specs'].vcpu} each  
                    **Memory:** {reader['specs'].memory_gb} GB each  
                    **Multi-AZ:** {'✅ Yes' if reader['multi_az'] else '❌ No'}  
                    **Cost per Reader:** ${reader['single_reader_monthly_cost']:,.0f}/month  
                    **Total Monthly Cost:** ${reader['total_monthly_cost']:,.0f}  
                    **Total Annual Cost:** ${reader['total_annual_cost']:,.0f}
                    """)
                else:
                    st.warning("**No read replicas recommended**")
                
                st.markdown("**Reasoning:**")
                st.write(reader['reasoning'])
            
            # Cost breakdown chart
            st.markdown("#### 💰 Cost Breakdown")
            
            cost_data = optimization['cost_analysis']['monthly_breakdown']
            
            fig = go.Figure(data=[go.Pie(
                labels=['Writer Instance', 'Reader Instances', 'Storage', 'Backup', 'Monitoring', 'Cross-AZ Transfer'],
                values=[cost_data['writer_instance'], cost_data['reader_instances'], 
                       cost_data['storage'], cost_data['backup'], cost_data['monitoring'], cost_data['cross_az_transfer']],
                hole=0.4,
                textinfo='label+percent+value',
                texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}'
            )])
            
            fig.update_layout(
                title=f"{env_name} Monthly Cost Distribution - Total: ${cost_data['total']:,.0f}",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Reserved Instance savings
            st.markdown("#### 💳 Reserved Instance Savings Opportunities")
            
            ri_1_year = optimization['cost_analysis']['reserved_instance_options']['1_year']
            ri_3_year = optimization['cost_analysis']['reserved_instance_options']['3_year']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1-Year Reserved Instances**")
                st.metric("Annual Savings", f"${ri_1_year['total_savings']:,.0f}", 
                         delta=f"{ri_1_year['discount_percentage']:.0f}% discount")
                st.write(f"Monthly cost: ${ri_1_year['monthly_cost']:,.0f}")
            
            with col2:
                st.markdown("**3-Year Reserved Instances**")
                st.metric("Total 3-Year Savings", f"${ri_3_year['total_savings']:,.0f}", 
                         delta=f"{ri_3_year['discount_percentage']:.0f}% discount")
                st.write(f"Monthly cost: ${ri_3_year['monthly_cost']:,.0f}")
            
            # Optimization opportunities
            st.markdown("#### 🎯 Cost Optimization Opportunities")
            
            opportunities = optimization['cost_analysis']['cost_optimization_opportunities']
            
            for opp in opportunities:
                impact_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}[opp['impact']]
                st.markdown(f"""
                {impact_color} **{opp['type']}** - {opp['impact']} Impact  
                {opp['description']}  
                💰 Potential savings: ${opp['potential_savings']:,.0f}/year  
                🔧 Implementation effort: {opp['implementation_effort']}
                """)
            
            # Workload analysis
            st.markdown("#### 📊 Workload Analysis")
            
            workload = optimization['workload_analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Intensity Metrics:**")
                st.progress(workload['cpu_intensity']/100, text=f"CPU Intensity: {workload['cpu_intensity']:.1f}%")
                st.progress(workload['memory_intensity']/100, text=f"Memory Intensity: {workload['memory_intensity']:.1f}%")
                st.progress(workload['io_intensity']/100, text=f"I/O Intensity: {workload['io_intensity']:.1f}%")
                st.progress(workload['connection_intensity']/100, text=f"Connection Intensity: {workload['connection_intensity']:.1f}%")
            
            with col2:
                st.markdown("**Workload Characteristics:**")
                st.write(f"**Type:** {workload['workload_type'].replace('_', ' ').title()}")
                st.write(f"**Complexity Score:** {workload['complexity_score']:.1f}/100")
                st.write(f"**Performance Tier:** {workload['performance_requirements'].replace('_', ' ').title()}")
                st.write(f"**Read Scaling Factor:** {workload['read_scaling_factor']:.1f}x")
            
            # Recommendations
            st.markdown("#### 💡 Optimization Recommendations")
            
            for rec in optimization['recommendations']:
                st.markdown(f"• {rec}")

# Example usage
def run_optimization_analysis():
    """Run the optimization analysis with sample data"""
    
    # Sample environment specifications
    sample_environments = {
        'Production': {
            'cpu_cores': 32,
            'ram_gb': 128,
            'storage_gb': 2000,
            'iops_requirement': 15000,
            'peak_connections': 1000,
            'workload_pattern': 'read_heavy',
            'read_write_ratio': 80,
            'environment_type': 'production',
            'daily_usage_hours': 24
        },
        'Staging': {
            'cpu_cores': 16,
            'ram_gb': 64,
            'storage_gb': 1000,
            'iops_requirement': 8000,
            'peak_connections': 300,
            'workload_pattern': 'balanced',
            'read_write_ratio': 70,
            'environment_type': 'staging',
            'daily_usage_hours': 16
        },
        'Development': {
            'cpu_cores': 8,
            'ram_gb': 32,
            'storage_gb': 500,
            'iops_requirement': 3000,
            'peak_connections': 100,
            'workload_pattern': 'write_heavy',
            'read_write_ratio': 40,
            'environment_type': 'development',
            'daily_usage_hours': 10
        }
    }
    
    # Initialize optimizer
    optimizer = OptimizedReaderWriterAnalyzer()
    
    # Run optimization
    optimization_results = optimizer.optimize_cluster_configuration(sample_environments)
    
    # Display results
    display_optimization_results(optimization_results)
    
    return optimization_results
@dataclass
class InstanceSpecs:
    """Instance specifications with performance metrics"""
    instance_class: str
    vcpu: int
    memory_gb: float
    network_performance: str
    ebs_bandwidth_mbps: int
    hourly_cost: float
    suitable_for: List[str]
    max_connections: int
    iops_capability: int

class OptimizedReaderWriterAnalyzer:
    """Advanced Reader/Writer optimization with intelligent recommendations"""
    
    def __init__(self):
        self.instance_specs = self._initialize_instance_specs()
        self.pricing_data = self._initialize_pricing_data()
        self.optimization_goal = 'balanced'
        self.consider_reserved_instances = True
        self.environment_priority = 'production_first'
        
    def _initialize_instance_specs(self) -> Dict[str, InstanceSpecs]:
        """Initialize comprehensive instance specifications"""
        return {
            # T3 Series - Burstable Performance
            'db.t3.micro': InstanceSpecs('db.t3.micro', 2, 1, 'Low to Moderate', 87, 0.0255, ['development', 'testing'], 87, 1000),
            'db.t3.small': InstanceSpecs('db.t3.small', 2, 2, 'Low to Moderate', 174, 0.051, ['development', 'testing'], 174, 2000),
            'db.t3.medium': InstanceSpecs('db.t3.medium', 2, 4, 'Low to Moderate', 347, 0.102, ['development', 'small_production'], 347, 3000),
            'db.t3.large': InstanceSpecs('db.t3.large', 2, 8, 'Low to Moderate', 695, 0.204, ['development', 'small_production'], 695, 4000),
            'db.t3.xlarge': InstanceSpecs('db.t3.xlarge', 4, 16, 'Low to Moderate', 695, 0.408, ['staging', 'medium_production'], 1000, 5000),
            'db.t3.2xlarge': InstanceSpecs('db.t3.2xlarge', 8, 32, 'Low to Moderate', 695, 0.816, ['staging', 'medium_production'], 1500, 6000),
            
            # R5 Series - Memory Optimized
            'db.r5.large': InstanceSpecs('db.r5.large', 2, 16, 'Up to 10 Gbps', 693, 0.24, ['memory_intensive', 'analytics'], 1000, 7500),
            'db.r5.xlarge': InstanceSpecs('db.r5.xlarge', 4, 32, 'Up to 10 Gbps', 1387, 0.48, ['memory_intensive', 'production'], 2000, 10000),
            'db.r5.2xlarge': InstanceSpecs('db.r5.2xlarge', 8, 64, 'Up to 10 Gbps', 2775, 0.96, ['large_production', 'analytics'], 3000, 15000),
            'db.r5.4xlarge': InstanceSpecs('db.r5.4xlarge', 16, 128, '10 Gbps', 4750, 1.92, ['large_production', 'high_memory'], 5000, 25000),
            'db.r5.8xlarge': InstanceSpecs('db.r5.8xlarge', 32, 256, '10 Gbps', 6800, 3.84, ['enterprise', 'high_performance'], 8000, 40000),
            'db.r5.12xlarge': InstanceSpecs('db.r5.12xlarge', 48, 384, '12 Gbps', 9500, 5.76, ['enterprise', 'very_high_memory'], 12000, 60000),
            'db.r5.16xlarge': InstanceSpecs('db.r5.16xlarge', 64, 512, '20 Gbps', 13600, 7.68, ['enterprise', 'extreme_performance'], 16000, 80000),
            'db.r5.24xlarge': InstanceSpecs('db.r5.24xlarge', 96, 768, '25 Gbps', 19000, 11.52, ['enterprise', 'maximum_performance'], 24000, 120000),
            
            # R6i Series - Latest Generation Memory Optimized
            'db.r6i.large': InstanceSpecs('db.r6i.large', 2, 16, 'Up to 12.5 Gbps', 1000, 0.252, ['memory_intensive', 'latest_gen'], 1200, 8000),
            'db.r6i.xlarge': InstanceSpecs('db.r6i.xlarge', 4, 32, 'Up to 12.5 Gbps', 2000, 0.504, ['production', 'latest_gen'], 2400, 12000),
            'db.r6i.2xlarge': InstanceSpecs('db.r6i.2xlarge', 8, 64, 'Up to 12.5 Gbps', 4000, 1.008, ['large_production', 'latest_gen'], 4000, 18000),
            'db.r6i.4xlarge': InstanceSpecs('db.r6i.4xlarge', 16, 128, '12.5 Gbps', 8000, 2.016, ['enterprise', 'latest_gen'], 6000, 30000),
            
            # C5 Series - Compute Optimized
            'db.c5.large': InstanceSpecs('db.c5.large', 2, 4, 'Up to 10 Gbps', 693, 0.192, ['compute_intensive', 'low_memory'], 1000, 5000),
            'db.c5.xlarge': InstanceSpecs('db.c5.xlarge', 4, 8, 'Up to 10 Gbps', 1387, 0.384, ['compute_intensive', 'oltp'], 2000, 8000),
            'db.c5.2xlarge': InstanceSpecs('db.c5.2xlarge', 8, 16, 'Up to 10 Gbps', 2775, 0.768, ['high_cpu', 'oltp'], 3000, 12000),
            'db.c5.4xlarge': InstanceSpecs('db.c5.4xlarge', 16, 32, '10 Gbps', 4750, 1.536, ['very_high_cpu', 'oltp'], 5000, 20000),
        }
    
    def _initialize_pricing_data(self) -> Dict:
        """Initialize comprehensive pricing data"""
        return {
            'us-east-1': {
                'reserved_1_year': {'discount': 0.35, 'upfront_ratio': 0.0},
                'reserved_3_year': {'discount': 0.55, 'upfront_ratio': 0.0},
                'spot': {'discount': 0.70, 'availability': 0.85},
                'multi_az_multiplier': 2.0,
                'cross_az_transfer_gb': 0.01,
                'backup_storage_gb': 0.095,
                'snapshot_storage_gb': 0.095
            }
        }
    
    def set_optimization_preferences(self, goal='balanced', consider_reserved=True, environment_priority='production_first'):
        """Set optimization preferences"""
        self.optimization_goal = goal
        self.consider_reserved_instances = consider_reserved
        self.environment_priority = environment_priority
    
    def optimize_cluster_configuration(self, environment_specs: Dict) -> Dict:
        """Generate optimized Reader/Writer recommendations with detailed cost analysis"""
        
        optimized_recommendations = {}
        
        for env_name, specs in environment_specs.items():
            optimization = self._optimize_single_environment(env_name, specs)
            optimized_recommendations[env_name] = optimization
        
        return optimized_recommendations
    
    def _optimize_single_environment(self, env_name: str, specs: Dict) -> Dict:
        """Optimize configuration for a single environment"""
        
        # Extract environment characteristics
        cpu_cores = specs.get('cpu_cores', 4)
        ram_gb = specs.get('ram_gb', 16)
        storage_gb = specs.get('storage_gb', 500)
        iops_requirement = specs.get('iops_requirement', 3000)
        peak_connections = specs.get('peak_connections', 100)
        workload_pattern = specs.get('workload_pattern', 'balanced')
        read_write_ratio = specs.get('read_write_ratio', 70)
        environment_type = specs.get('environment_type', 'production')
        daily_usage_hours = specs.get('daily_usage_hours', 24)
        
        # Analyze workload characteristics
        workload_analysis = self._analyze_workload_characteristics(
            cpu_cores, ram_gb, iops_requirement, peak_connections, 
            workload_pattern, read_write_ratio
        )
        
        # Optimize Writer configuration
        writer_optimization = self._optimize_writer_instance(workload_analysis, environment_type)
        
        # Optimize Reader configuration
        reader_optimization = self._optimize_reader_configuration(
            writer_optimization, workload_analysis, environment_type
        )
        
        # Calculate comprehensive costs
        cost_analysis = self._calculate_comprehensive_costs(
            writer_optimization, reader_optimization, storage_gb, 
            iops_requirement, environment_type, daily_usage_hours
        )
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            workload_analysis, writer_optimization, reader_optimization, cost_analysis
        )
        
        return {
            'environment_name': env_name,
            'environment_type': environment_type,
            'workload_analysis': workload_analysis,
            'writer_optimization': writer_optimization,
            'reader_optimization': reader_optimization,
            'cost_analysis': cost_analysis,
            'recommendations': recommendations,
            'optimization_score': self._calculate_optimization_score(
                workload_analysis, writer_optimization, reader_optimization, cost_analysis
            )
        }
    
    def _analyze_workload_characteristics(self, cpu_cores: int, ram_gb: int, 
                                        iops_requirement: int, peak_connections: int,
                                        workload_pattern: str, read_write_ratio: int) -> Dict:
        """Analyze workload characteristics to determine optimal configuration"""
        
        # Calculate workload intensity
        cpu_intensity = min(100, (cpu_cores / 64) * 100)
        memory_intensity = min(100, (ram_gb / 768) * 100)
        io_intensity = min(100, (iops_requirement / 80000) * 100)
        connection_intensity = min(100, (peak_connections / 16000) * 100)
        
        # Determine workload type
        if workload_pattern == 'read_heavy' or read_write_ratio >= 80:
            workload_type = 'read_heavy'
            read_scaling_factor = 2.0
        elif workload_pattern == 'write_heavy' or read_write_ratio <= 30:
            workload_type = 'write_heavy'
            read_scaling_factor = 0.5
        elif workload_pattern == 'analytics':
            workload_type = 'analytics'
            read_scaling_factor = 3.0
        else:
            workload_type = 'balanced'
            read_scaling_factor = 1.2
        
        # Calculate complexity score
        complexity_score = (cpu_intensity + memory_intensity + io_intensity + connection_intensity) / 4
        
        # Calculate optimal reader count
        base_readers = {
            'read_heavy': 3,
            'analytics': 2,
            'balanced': 1,
            'write_heavy': 0
        }
        
        complexity_adjustment = int(complexity_score / 30)
        connection_adjustment = int(peak_connections / 2000)
        optimal_reader_count = base_readers.get(workload_type, 1) + complexity_adjustment + connection_adjustment
        optimal_reader_count = min(5, max(0, optimal_reader_count))
        
        # Determine performance requirements
        avg_intensity = (cpu_intensity + memory_intensity + io_intensity) / 3
        if avg_intensity >= 80:
            performance_requirements = 'high_performance'
        elif avg_intensity >= 60:
            performance_requirements = 'medium_performance'
        elif avg_intensity >= 40:
            performance_requirements = 'standard_performance'
        else:
            performance_requirements = 'basic_performance'
        
        return {
            'cpu_intensity': cpu_intensity,
            'memory_intensity': memory_intensity,
            'io_intensity': io_intensity,
            'connection_intensity': connection_intensity,
            'complexity_score': complexity_score,
            'workload_type': workload_type,
            'read_scaling_factor': read_scaling_factor,
            'recommended_reader_count': optimal_reader_count,
            'performance_requirements': performance_requirements
        }
    
    def _optimize_writer_instance(self, workload_analysis: Dict, environment_type: str) -> Dict:
        """Optimize Writer instance selection"""
        
        performance_req = workload_analysis['performance_requirements']
        complexity_score = workload_analysis['complexity_score']
        
        # Filter suitable instances
        suitable_instances = []
        
        for instance_class, specs in self.instance_specs.items():
            if self._is_instance_suitable_for_writer(specs, performance_req, environment_type, complexity_score):
                suitable_instances.append((instance_class, specs))
        
        # Score instances
        scored_instances = []
        for instance_class, specs in suitable_instances:
            score = self._score_writer_instance(specs, workload_analysis, environment_type)
            scored_instances.append((score, instance_class, specs))
        
        # Sort by score
        scored_instances.sort(reverse=True)
        
        # Get top recommendation
        if scored_instances:
            score, instance_class, specs = scored_instances[0]
            
            return {
                'instance_class': instance_class,
                'specs': specs,
                'score': score,
                'multi_az': environment_type in ['production', 'staging'],
                'reasoning': self._generate_writer_reasoning(specs, workload_analysis, environment_type),
                'monthly_cost': self._calculate_instance_monthly_cost(specs, environment_type),
                'annual_cost': self._calculate_instance_monthly_cost(specs, environment_type) * 12
            }
        else:
            # Fallback
            fallback_instance = 'db.r5.large'
            fallback_specs = self.instance_specs[fallback_instance]
            
            return {
                'instance_class': fallback_instance,
                'specs': fallback_specs,
                'score': 50.0,
                'multi_az': True,
                'reasoning': 'Fallback recommendation',
                'monthly_cost': self._calculate_instance_monthly_cost(fallback_specs, environment_type),
                'annual_cost': self._calculate_instance_monthly_cost(fallback_specs, environment_type) * 12
            }
    
    def _is_instance_suitable_for_writer(self, specs: InstanceSpecs, performance_req: str, 
                                       environment_type: str, complexity_score: float) -> bool:
        """Check if instance is suitable for writer role"""
        
        performance_minimums = {
            'high_performance': {'min_vcpu': 16, 'min_memory': 64, 'min_iops': 20000},
            'medium_performance': {'min_vcpu': 8, 'min_memory': 32, 'min_iops': 10000},
            'standard_performance': {'min_vcpu': 4, 'min_memory': 16, 'min_iops': 5000},
            'basic_performance': {'min_vcpu': 2, 'min_memory': 8, 'min_iops': 2000}
        }
        
        minimums = performance_minimums.get(performance_req, performance_minimums['standard_performance'])
        
        if (specs.vcpu < minimums['min_vcpu'] or 
            specs.memory_gb < minimums['min_memory'] or 
            specs.iops_capability < minimums['min_iops']):
            return False
        
        if environment_type == 'production':
            if specs.vcpu < 4 or specs.memory_gb < 16:
                return False
            if 't3' in specs.instance_class and complexity_score > 50:
                return False
        
        return True
    
    def _score_writer_instance(self, specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> float:
        """Score writer instance based on multiple criteria"""
        
        score = 0.0
        
        # Performance scoring (40% weight)
        performance_score = min(100, (specs.vcpu * 10 + specs.memory_gb / 4 + specs.iops_capability / 500))
        score += performance_score * 0.4
        
        # Cost efficiency scoring (30% weight)
        cost_per_vcpu = specs.hourly_cost / specs.vcpu
        cost_per_gb_memory = specs.hourly_cost / specs.memory_gb
        cost_efficiency = max(0, 100 - (cost_per_vcpu * 100 + cost_per_gb_memory * 10))
        score += cost_efficiency * 0.3
        
        # Suitability scoring (20% weight)
        suitability_score = 60
        if environment_type in specs.suitable_for:
            suitability_score = 100
        elif any(env in specs.suitable_for for env in ['production', 'large_production', 'enterprise']):
            suitability_score = 80
        score += suitability_score * 0.2
        
        # Network performance scoring (10% weight)
        network_score = 60
        if '25 Gbps' in specs.network_performance:
            network_score = 100
        elif '20 Gbps' in specs.network_performance:
            network_score = 90
        elif '12' in specs.network_performance:
            network_score = 80
        elif '10 Gbps' in specs.network_performance:
            network_score = 70
        score += network_score * 0.1
        
        return min(100, score)
    
    def _optimize_reader_configuration(self, writer_optimization: Dict, workload_analysis: Dict, environment_type: str) -> Dict:
        """Optimize Reader configuration"""
        
        recommended_count = workload_analysis['recommended_reader_count']
        
        if recommended_count == 0:
            return {
                'count': 0,
                'instance_class': None,
                'specs': None,
                'single_reader_monthly_cost': 0,
                'total_monthly_cost': 0,
                'total_annual_cost': 0,
                'multi_az': False,
                'reasoning': 'No read replicas needed for this workload pattern'
            }
        
        # Determine optimal reader instance size
        writer_specs = writer_optimization['specs']
        reader_instance_class = self._calculate_optimal_reader_size(writer_specs, workload_analysis, environment_type)
        reader_specs = self.instance_specs[reader_instance_class]
        
        # Calculate costs
        single_reader_monthly_cost = self._calculate_instance_monthly_cost(reader_specs, environment_type, is_reader=True)
        total_monthly_cost = single_reader_monthly_cost * recommended_count
        total_annual_cost = total_monthly_cost * 12
        
        return {
            'count': recommended_count,
            'instance_class': reader_instance_class,
            'specs': reader_specs,
            'single_reader_monthly_cost': single_reader_monthly_cost,
            'total_monthly_cost': total_monthly_cost,
            'total_annual_cost': total_annual_cost,
            'multi_az': environment_type == 'production',
            'reasoning': self._generate_reader_reasoning(recommended_count, reader_specs, workload_analysis)
        }
    
    def _calculate_optimal_reader_size(self, writer_specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> str:
        """Calculate optimal reader instance size"""
        
        workload_type = workload_analysis['workload_type']
        read_scaling_factor = workload_analysis['read_scaling_factor']
        
        if workload_type == 'read_heavy' or workload_type == 'analytics':
            target_vcpu = int(writer_specs.vcpu * read_scaling_factor)
            target_memory = writer_specs.memory_gb * read_scaling_factor
        else:
            target_vcpu = max(2, int(writer_specs.vcpu * 0.7))
            target_memory = writer_specs.memory_gb * 0.7
        
        # Find best matching instance
        best_match = None
        best_score = 0
        
        for instance_class, specs in self.instance_specs.items():
            if specs.vcpu >= target_vcpu * 0.8 and specs.memory_gb >= target_memory * 0.8:
                size_match_score = 100 - abs(specs.vcpu - target_vcpu) * 5 - abs(specs.memory_gb - target_memory) * 2
                cost_efficiency_score = 100 - (specs.hourly_cost * 10)
                overall_score = size_match_score * 0.7 + cost_efficiency_score * 0.3
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_match = instance_class
        
        return best_match or 'db.r5.large'
    
    def _calculate_instance_monthly_cost(self, specs: InstanceSpecs, environment_type: str, is_reader: bool = False) -> float:
        """Calculate monthly cost for an instance"""
        
        base_hourly_cost = specs.hourly_cost
        
        # Apply Multi-AZ multiplier if needed
        if environment_type in ['production', 'staging'] and not is_reader:
            base_hourly_cost *= self.pricing_data['us-east-1']['multi_az_multiplier']
        
        # Calculate monthly cost (730 hours per month)
        monthly_cost = base_hourly_cost * 730
        
        return monthly_cost
    
    def _calculate_comprehensive_costs(self, writer_optimization: Dict, reader_optimization: Dict,
                                     storage_gb: int, iops_requirement: int, environment_type: str,
                                     daily_usage_hours: int) -> Dict:
        """Calculate comprehensive cost analysis"""
        
        # Instance costs
        writer_monthly_cost = writer_optimization['monthly_cost']
        reader_monthly_cost = reader_optimization['total_monthly_cost']
        total_instance_monthly_cost = writer_monthly_cost + reader_monthly_cost
        
        # Storage costs
        storage_costs = self._calculate_storage_costs(storage_gb, iops_requirement, environment_type)
        
        # Additional costs
        backup_monthly_cost = storage_gb * 0.095
        monitoring_monthly_cost = 50 if environment_type == 'production' else 20
        cross_az_monthly_cost = 0
        
        if reader_optimization['count'] > 0:
            estimated_cross_az_gb = storage_gb * 0.1
            cross_az_monthly_cost = estimated_cross_az_gb * 0.01
        
        total_monthly_cost = (total_instance_monthly_cost + storage_costs['total_monthly'] + 
                            backup_monthly_cost + monitoring_monthly_cost + cross_az_monthly_cost)
        
        # Reserved Instance calculations
        reserved_1_year = self._calculate_reserved_instance_savings(total_instance_monthly_cost, 1)
        reserved_3_year = self._calculate_reserved_instance_savings(total_instance_monthly_cost, 3)
        
        return {
            'monthly_breakdown': {
                'writer_instance': writer_monthly_cost,
                'reader_instances': reader_monthly_cost,
                'storage': storage_costs['total_monthly'],
                'backup': backup_monthly_cost,
                'monitoring': monitoring_monthly_cost,
                'cross_az_transfer': cross_az_monthly_cost,
                'total': total_monthly_cost
            },
            'annual_breakdown': {
                'writer_instance': writer_monthly_cost * 12,
                'reader_instances': reader_monthly_cost * 12,
                'storage': storage_costs['total_monthly'] * 12,
                'backup': backup_monthly_cost * 12,
                'monitoring': monitoring_monthly_cost * 12,
                'cross_az_transfer': cross_az_monthly_cost * 12,
                'total': total_monthly_cost * 12
            },
            'storage_details': storage_costs,
            'reserved_instance_options': {
                '1_year': reserved_1_year,
                '3_year': reserved_3_year
            },
            'cost_optimization_opportunities': []
        }
    
    def _calculate_storage_costs(self, storage_gb: int, iops_requirement: int, environment_type: str) -> Dict:
        """Calculate detailed storage costs"""
        
        if iops_requirement > 16000:
            storage_type = 'io2'
            base_cost_per_gb = 0.125
            iops_cost_per_iops = 0.065
        elif iops_requirement > 3000:
            storage_type = 'gp3'
            base_cost_per_gb = 0.08
            additional_iops = max(0, iops_requirement - 3000)
            iops_cost_per_iops = 0.005 if additional_iops > 0 else 0
        else:
            storage_type = 'gp3'
            base_cost_per_gb = 0.08
            iops_cost_per_iops = 0
            additional_iops = 0
        
        base_storage_cost = storage_gb * base_cost_per_gb
        iops_cost = (additional_iops if storage_type == 'gp3' else iops_requirement) * iops_cost_per_iops
        total_monthly_cost = base_storage_cost + iops_cost
        
        return {
            'storage_type': storage_type,
            'storage_gb': storage_gb,
            'iops_provisioned': iops_requirement,
            'base_storage_cost': base_storage_cost,
            'iops_cost': iops_cost,
            'total_monthly': total_monthly_cost,
            'cost_per_gb': base_cost_per_gb,
            'cost_per_iops': iops_cost_per_iops
        }
    
    def _calculate_reserved_instance_savings(self, monthly_instance_cost: float, years: int) -> Dict:
        """Calculate Reserved Instance savings"""
        
        if years == 1:
            discount = self.pricing_data['us-east-1']['reserved_1_year']['discount']
        else:
            discount = self.pricing_data['us-east-1']['reserved_3_year']['discount']
        
        annual_on_demand = monthly_instance_cost * 12
        annual_reserved = annual_on_demand * (1 - discount)
        total_on_demand = annual_on_demand * years
        total_reserved = annual_reserved * years
        total_savings = total_on_demand - total_reserved
        
        return {
            'term_years': years,
            'discount_percentage': discount * 100,
            'annual_cost': annual_reserved,
            'total_cost': total_reserved,
            'total_savings': total_savings,
            'monthly_cost': annual_reserved / 12
        }
    
    def _generate_optimization_recommendations(self, workload_analysis: Dict, writer_optimization: Dict,
                                             reader_optimization: Dict, cost_analysis: Dict) -> List[str]:
        """Generate comprehensive optimization recommendations"""
        
        recommendations = []
        
        if workload_analysis['complexity_score'] > 80:
            recommendations.append("Consider upgrading to latest generation instances (R6i) for better price/performance")
        
        total_annual_cost = cost_analysis['annual_breakdown']['total']
        if total_annual_cost > 50000:
            recommendations.append("Evaluate Reserved Instances for significant cost savings on predictable workloads")
        
        if reader_optimization['count'] > 1:
            recommendations.append("Implement Aurora Auto Scaling to optimize reader count based on actual demand")
        
        recommendations.append("Set up Enhanced Monitoring and Performance Insights for optimization opportunities")
        recommendations.append("Configure CloudWatch alarms for CPU, memory, and connection metrics")
        
        return recommendations
    
    def _calculate_optimization_score(self, workload_analysis: Dict, writer_optimization: Dict,
                                    reader_optimization: Dict, cost_analysis: Dict) -> float:
        """Calculate overall optimization score"""
        
        performance_score = min(100, writer_optimization['score'])
        cost_per_vcpu = cost_analysis['monthly_breakdown']['total'] / writer_optimization['specs'].vcpu
        cost_efficiency_score = max(0, 100 - (cost_per_vcpu / 10))
        
        config_score = 80
        if reader_optimization['count'] > 0 and workload_analysis['workload_type'] in ['read_heavy', 'analytics']:
            config_score += 15
        elif reader_optimization['count'] == 0 and workload_analysis['workload_type'] == 'write_heavy':
            config_score += 15
        
        overall_score = (performance_score * 0.4 + cost_efficiency_score * 0.4 + config_score * 0.2)
        
        return min(100, overall_score)
    
    def _generate_writer_reasoning(self, specs: InstanceSpecs, workload_analysis: Dict, environment_type: str) -> str:
        """Generate reasoning for writer instance recommendation"""
        
        reasoning_parts = []
        
        if workload_analysis['complexity_score'] > 70:
            reasoning_parts.append(f"High complexity workload requires {specs.vcpu} vCPUs and {specs.memory_gb}GB memory")
        else:
            reasoning_parts.append(f"Balanced configuration with {specs.vcpu} vCPUs and {specs.memory_gb}GB memory")
        
        if environment_type == 'production':
            reasoning_parts.append("Production-grade instance with high availability features")
        else:
            reasoning_parts.append(f"Cost-optimized for {environment_type} environment")
        
        if '25 Gbps' in specs.network_performance:
            reasoning_parts.append("Enhanced networking for high-throughput workloads")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_reader_reasoning(self, count: int, specs: InstanceSpecs, workload_analysis: Dict) -> str:
        """Generate reasoning for reader configuration"""
        
        if count == 0:
            return "No read replicas needed for write-heavy or low-complexity workloads"
        
        workload_type = workload_analysis['workload_type']
        
        reasoning_parts = [
            f"{count} read replica{'s' if count > 1 else ''} recommended for {workload_type} workload",
            f"Each reader: {specs.vcpu} vCPUs, {specs.memory_gb}GB memory"
        ]
        
        if workload_type == 'read_heavy':
            reasoning_parts.append("Multiple readers will distribute read load effectively")
        elif workload_type == 'analytics':
            reasoning_parts.append("Dedicated readers for analytical queries to avoid impact on primary")
        
        return ". ".join(reasoning_parts) + "."
# Integration with your existing Streamlit app
def integrate_with_existing_app():
    """Integration instructions for your existing app"""
    
    st.markdown("## 🔗 Integration Instructions")
    
    st.code("""
    # Add this to your show_enhanced_environment_setup_with_cluster_config() function:
    
    if st.button("🚀 Run Optimization Analysis", type="primary"):
        optimizer = OptimizedReaderWriterAnalyzer()
        optimization_results = optimizer.optimize_cluster_configuration(st.session_state.environment_specs)
        st.session_state.optimization_results = optimization_results
        display_optimization_results(optimization_results)
    
    # Add this to your results dashboard:
    
    if hasattr(st.session_state, 'optimization_results'):
        display_optimization_results(st.session_state.optimization_results)
    """)

def show_optimized_recommendations():
    """Show optimized Reader/Writer recommendations"""
    
    st.markdown("## 🚀 AI-Optimized Reader/Writer Recommendations")
    
    if not st.session_state.environment_specs:
        st.warning("⚠️ Please configure environments first.")
        return
    
    # Configuration options
    st.markdown("### ⚙️ Optimization Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        optimization_goal = st.selectbox(
            "Optimization Goal",
            ["Cost Efficiency", "Performance", "Balanced"],
            index=2
        )
    
    with col2:
        consider_reserved_instances = st.checkbox(
            "Include Reserved Instance Analysis",
            value=True
        )
    
    with col3:
        environment_priority = st.selectbox(
            "Environment Priority",
            ["Production First", "All Equal", "Cost First"],
            index=0
        )
    
    # Run optimization button
    if st.button("🧠 Generate AI Recommendations", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing workloads and optimizing configurations..."):
            
            # Initialize optimizer
            optimizer = OptimizedReaderWriterAnalyzer()
            
            # Apply optimization settings
            optimizer.set_optimization_preferences(
                goal=optimization_goal.lower().replace(" ", "_"),
                consider_reserved=consider_reserved_instances,
                environment_priority=environment_priority.lower().replace(" ", "_")
            )
            
            # Run optimization
            optimization_results = optimizer.optimize_cluster_configuration(st.session_state.environment_specs)
            
            # Store results
            st.session_state.optimization_results = optimization_results
            
            st.success("✅ Optimization complete!")
    
    # Display results if available
    if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
        display_optimization_results(st.session_state.optimization_results)
        
        # Export recommendations
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export to CSV", use_container_width=True):
                csv_data = export_recommendations_to_csv(st.session_state.optimization_results)
                st.download_button(
                    label="📥 Download Recommendations CSV",
                    data=csv_data,
                    file_name=f"reader_writer_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📄 Generate Detailed Report", use_container_width=True):
                pdf_data = generate_optimization_report_pdf(st.session_state.optimization_results)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"reader_writer_optimization_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )

def show_enhanced_environment_analysis():
    """Show enhanced environment analysis with Writer/Reader details"""
    
    st.markdown("### 🏢 Enhanced Environment Analysis")
    
    recommendations = st.session_state.enhanced_recommendations
    environment_specs = st.session_state.environment_specs
    
    # Environment comparison with cluster details
    env_comparison_data = []
    
    for env_name, rec in recommendations.items():
        specs = environment_specs[env_name]
        
        # Writer configuration
        writer_config = f"{rec['writer']['instance_class']} ({'Multi-AZ' if rec['writer']['multi_az'] else 'Single-AZ'})"
        
        # Reader configuration
        reader_count = rec['readers']['count']
        if reader_count > 0:
            reader_config = f"{reader_count} x {rec['readers']['instance_class']}"
        else:
            reader_config = "No readers"
        
        env_comparison_data.append({
            'Environment': env_name,
            'Type': rec['environment_type'].title(),
            'Current Resources': f"{specs['cpu_cores']} cores, {specs['ram_gb']} GB RAM",
            'Writer Instance': writer_config,
            'Read Replicas': reader_config,
            'Storage': f"{rec['storage']['size_gb']} GB {rec['storage']['type'].upper()}",
            'Workload Pattern': f"{rec['workload_pattern']} ({rec['read_write_ratio']}% reads)"
        })
    
    env_df = pd.DataFrame(env_comparison_data)
    st.dataframe(env_df, use_container_width=True)
    
    # Detailed environment insights
    st.markdown("#### 💡 Environment Insights")
    
    for env_name, rec in recommendations.items():
        with st.expander(f"🔍 {env_name} Environment Details"):
            specs = environment_specs[env_name]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Current Configuration**")
                st.write(f"CPU Cores: {specs['cpu_cores']}")
                st.write(f"RAM: {specs['ram_gb']} GB")
                st.write(f"Storage: {specs['storage_gb']} GB")
                st.write(f"IOPS Requirement: {specs.get('iops_requirement', 'N/A')}")
                st.write(f"Peak Connections: {specs.get('peak_connections', 'N/A')}")
            
            with col2:
                st.markdown("**Writer Configuration**")
                writer = rec['writer']
                st.write(f"Instance: {writer['instance_class']}")
                st.write(f"Multi-AZ: {'Yes' if writer['multi_az'] else 'No'}")
                st.write(f"CPU Cores: {writer['cpu_cores']}")
                st.write(f"RAM: {writer['ram_gb']} GB")
                
                st.markdown("**Reader Configuration**")
                readers = rec['readers']
                if readers['count'] > 0:
                    st.write(f"Count: {readers['count']}")
                    st.write(f"Instance: {readers['instance_class']}")
                    st.write(f"Multi-AZ: {'Yes' if readers['multi_az'] else 'No'}")
                else:
                    st.write("No read replicas")
            
            with col3:
                st.markdown("**Storage Configuration**")
                storage = rec['storage']
                st.write(f"Size: {storage['size_gb']} GB")
                st.write(f"Type: {storage['type'].upper()}")
                st.write(f"IOPS: {storage['iops']:,}")
                st.write(f"Encrypted: {'Yes' if storage['encrypted'] else 'No'}")
                st.write(f"Backup Retention: {storage['backup_retention_days']} days")
                
                st.markdown("**Workload Characteristics**")
                st.write(f"Pattern: {rec['workload_pattern']}")
                st.write(f"Read/Write Ratio: {rec['read_write_ratio']}% reads")
                st.write(f"Peak Connections: {rec['connections']}")
            
            # Optimization recommendations
            st.markdown("**💡 Optimization Notes**")
            
            if rec['environment_type'] == 'production':
                st.success("✅ Production-grade configuration with high availability")
                if readers['count'] > 0:
                    st.info(f"📊 {readers['count']} read replicas will help distribute read load")
            elif rec['environment_type'] == 'development':
                st.info("💡 Cost-optimized configuration for development")
                if readers['count'] == 0:
                    st.info("💰 No read replicas to minimize development costs")
            
            if rec['workload_pattern'] == 'read_heavy' and readers['count'] > 0:
                st.success(f"📈 Read-heavy workload well-suited for {readers['count']} read replicas")
            elif rec['workload_pattern'] == 'read_heavy' and readers['count'] == 0:
                st.warning("⚠️ Read-heavy workload might benefit from read replicas")
            
            if storage['type'] == 'io2':
                st.info("⚡ High-performance io2 storage for demanding IOPS requirements")
            elif storage['type'] == 'gp3':
                st.info("⚖️ Balanced gp3 storage for general-purpose workloads")

import asyncio
import streamlit as st
import anthropic
import boto3
from typing import Dict, Optional
import json

# ADD THIS CLASS to your streamlit_app.py file (put it near the top with other classes):

class EnhancedAWSPricingAPI:
    """Enhanced AWS Pricing API with Writer/Reader and Aurora support"""
    
    def __init__(self):
        self.base_url = "https://pricing.us-east-1.amazonaws.com"
        self.cache = {}
        
    def calculate_cluster_cost(self, region: str, engine: str, 
                          writer_instance: str, writer_multi_az: bool,
                          reader_instances: List[Tuple[str, bool]]) -> float:
        """Calculate total cost for writer and readers"""
        total_cost = 0
    
        # Writer cost
        writer_pricing = self.get_rds_pricing(region, engine, writer_instance, writer_multi_az)
        total_cost += writer_pricing['hourly']
    
        for reader_instance, multi_az in reader_instances:
            reader_pricing = self.get_rds_pricing(region, engine, reader_instance, multi_az)
            total_cost += reader_pricing['hourly']
    
        return total_cost  # Fixed indentation - moved outside the loop
    
        
    def get_rds_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool = False) -> Dict:
        """Get RDS pricing for specific instance with Multi-AZ support"""
        cache_key = f"{region}_{engine}_{instance_class}_{multi_az}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Enhanced pricing data with Multi-AZ and Aurora support
        pricing_data = {
            'us-east-1': {
                'postgres': {
                    'db.t3.micro': {'hourly': 0.0255, 'hourly_multi_az': 0.051, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.small': {'hourly': 0.051, 'hourly_multi_az': 0.102, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.medium': {'hourly': 0.102, 'hourly_multi_az': 0.204, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.large': {'hourly': 0.204, 'hourly_multi_az': 0.408, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.xlarge': {'hourly': 0.408, 'hourly_multi_az': 0.816, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.24, 'hourly_multi_az': 0.48, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 0.48, 'hourly_multi_az': 0.96, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 0.96, 'hourly_multi_az': 1.92, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.4xlarge': {'hourly': 1.92, 'hourly_multi_az': 3.84, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.8xlarge': {'hourly': 3.84, 'hourly_multi_az': 7.68, 'storage_gb': 0.115, 'iops_gb': 0.10},
                },
                'aurora-postgresql': {
                    'db.r5.large': {'hourly': 0.29, 'hourly_multi_az': 0.29, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.xlarge': {'hourly': 0.58, 'hourly_multi_az': 0.58, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.2xlarge': {'hourly': 1.16, 'hourly_multi_az': 1.16, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.4xlarge': {'hourly': 2.32, 'hourly_multi_az': 2.32, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.8xlarge': {'hourly': 4.64, 'hourly_multi_az': 4.64, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.12xlarge': {'hourly': 6.96, 'hourly_multi_az': 6.96, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.16xlarge': {'hourly': 9.28, 'hourly_multi_az': 9.28, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.24xlarge': {'hourly': 13.92, 'hourly_multi_az': 13.92, 'storage_gb': 0.10, 'io_request': 0.20},
                },
                'oracle-ee': {
                    'db.t3.medium': {'hourly': 0.408, 'hourly_multi_az': 0.816, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.96, 'hourly_multi_az': 1.92, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 1.92, 'hourly_multi_az': 3.84, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 3.84, 'hourly_multi_az': 7.68, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.4xlarge': {'hourly': 7.68, 'hourly_multi_az': 15.36, 'storage_gb': 0.115, 'iops_gb': 0.10},
                },
                'mysql': {
                    'db.t3.micro': {'hourly': 0.0255, 'hourly_multi_az': 0.051, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.small': {'hourly': 0.051, 'hourly_multi_az': 0.102, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.medium': {'hourly': 0.102, 'hourly_multi_az': 0.204, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.large': {'hourly': 0.204, 'hourly_multi_az': 0.408, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.24, 'hourly_multi_az': 0.48, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 0.48, 'hourly_multi_az': 0.96, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 0.96, 'hourly_multi_az': 1.92, 'storage_gb': 0.115, 'iops_gb': 0.10},
                },
                'aurora-mysql': {
                    'db.r5.large': {'hourly': 0.29, 'hourly_multi_az': 0.29, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.xlarge': {'hourly': 0.58, 'hourly_multi_az': 0.58, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.2xlarge': {'hourly': 1.16, 'hourly_multi_az': 1.16, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.4xlarge': {'hourly': 2.32, 'hourly_multi_az': 2.32, 'storage_gb': 0.10, 'io_request': 0.20},
                    'db.r5.8xlarge': {'hourly': 4.64, 'hourly_multi_az': 4.64, 'storage_gb': 0.10, 'io_request': 0.20},
                }
            }
        }
        
        engine_pricing = pricing_data.get(region, {}).get(engine, {})
        instance_pricing = engine_pricing.get(instance_class, {
            'hourly': 0.5, 
            'hourly_multi_az': 1.0, 
            'storage_gb': 0.115, 
            'iops_gb': 0.10,
            'io_request': 0.20
        })
        
        # Select appropriate pricing based on Multi-AZ
        if multi_az and 'aurora' not in engine:
            hourly_cost = instance_pricing['hourly_multi_az']
        else:
            hourly_cost = instance_pricing['hourly']
        
        result = {
            'hourly': hourly_cost,
            'storage_gb': instance_pricing['storage_gb'],
            'iops_gb': instance_pricing.get('iops_gb', 0.10),
            'io_request': instance_pricing.get('io_request', 0.20),
            'is_aurora': 'aurora' in engine,
            'multi_az': multi_az
        }
        
        self.cache[cache_key] = result
        return result


# ALSO ADD this if MigrationAnalyzer class is missing:

class MigrationAnalyzer:
    """Basic migration analyzer for standard environment configurations"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.pricing_api = EnhancedAWSPricingAPI()
        self.anthropic_api_key = anthropic_api_key
           
    def calculate_instance_recommendations(self, environment_specs: Dict) -> Dict:
        """Calculate AWS instance recommendations for environments"""
        
        recommendations = {}
        
        for env_name, specs in environment_specs.items():
            cpu_cores = specs['cpu_cores']
            ram_gb = specs['ram_gb']
            storage_gb = specs['storage_gb']
            
            # Determine environment type
            environment_type = self._categorize_environment(env_name)
            
            # Calculate instance class
            instance_class = self._calculate_instance_class(cpu_cores, ram_gb, environment_type)
            
            # Multi-AZ recommendation
            multi_az = environment_type in ['production', 'staging']
                                  
            recommendations[env_name] = {
                'environment_type': environment_type,
                'instance_class': instance_class,
                'cpu_cores': cpu_cores,
                'ram_gb': ram_gb,
                'storage_gb': storage_gb,
                'multi_az': multi_az,
                'daily_usage_hours': specs.get('daily_usage_hours', 24),
                'peak_connections': specs.get('peak_connections', 100)
            }
        
        return recommendations
        for env_name, specs in environment_specs.items():
            
            # Calculate reader configuration
            reader_count = self._calculate_reader_count(
                specs['read_write_ratio'], 
                specs['workload_pattern'],
                environment_type
            )
            reader_instance = self._calculate_reader_instance_class(
                instance_class, environment_type
            )
            
            recommendations[env_name] = {
            'reader_count': reader_count,
                'reader_instance_class': reader_instance,
                'reader_multi_az': multi_az if environment_type == 'production' else False
            }
     
    def _calculate_reader_count(self, read_ratio: int, workload_pattern: str, env_type: str) -> int:
        """Calculate number of read replicas needed"""
        if env_type != 'production':
            return 0  # Only production gets readers
            
        if workload_pattern == 'read_heavy' and read_ratio >= 70:
            return 2
        elif workload_pattern == 'read_heavy' and read_ratio >= 50:
            return 1
        elif workload_pattern == 'mixed' and read_ratio >= 60:
            return 1
        return 0
    
    def _calculate_reader_instance_class(self, writer_instance: str, env_type: str) -> str:
        """Determine appropriate reader instance class"""
        if env_type != 'production':
            return "N/A"
        
        # Readers typically match writer class or one size smaller
        instance_map = {
            'db.r5.8xlarge': 'db.r5.4xlarge',
            'db.r5.4xlarge': 'db.r5.2xlarge',
            'db.r5.2xlarge': 'db.r5.xlarge',
            'db.r5.xlarge': 'db.r5.large',
            'db.r5.large': 'db.t3.large',
            'db.t3.large': 'db.t3.medium'
        }
        return instance_map.get(writer_instance, writer_instance)

# Update the StreamlitClaudeAIAnalyzer class
class StreamlitClaudeAIAnalyzer:
    # ... existing code ...
    
    def _prepare_migration_context(self, cost_analysis: Dict, migration_params: Dict, recommendations: Dict) -> str:
        """Enhanced to include reader/writer configurations"""
        context = super()._prepare_migration_context(cost_analysis, migration_params)
        
        # Add reader/writer details
        context += "\n\nREADER/WRITER CONFIGURATIONS:"
        for env_name, rec in recommendations.items():
            writer = rec['writer_instance_class']
            readers = rec['reader_count']
            reader_type = rec['reader_instance_class']
            
            context += f"\n- {env_name}: Writer: {writer}, Readers: {readers}x{reader_type}"
        
        return context

# Update the show_enhanced_environment_analysis function
def show_enhanced_environment_analysis():
    """Enhanced to show AI recommendations for reader/writer"""
    # ... existing code ...
    
    # Add AI recommendations section
    st.markdown("## 🤖 AI Optimization Recommendations")
    
    if "ai_insights" in st.session_state:
        ai_data = st.session_state.ai_insights
        if 'key_recommendations' in ai_data:
            for rec in ai_data['key_recommendations']:
                st.info(f"• {rec}")
                
            if 'reader_writer_analysis' in ai_data:
                st.subheader("Reader/Writer Optimization")
                st.write(ai_data['reader_writer_analysis'])
        else:
            st.warning("No specific reader/writer recommendations available")
    else:
        st.info("Run analysis to generate AI recommendations")

# Update the generate_ai_insights_sync method
def generate_ai_insights_sync(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
    """Enhanced to include reader/writer analysis"""
    # ... existing code ...
    
    # Add reader/writer specific analysis
    ai_response += "\n\nREADER/WRITER ANALYSIS:\n"
    ai_response += self._generate_reader_writer_analysis(recommendations)
    
    # Add to structured insights
    structured_insights['reader_writer_analysis'] = self._generate_reader_writer_analysis(recommendations)
    
    return structured_insights

def _generate_reader_writer_analysis(self, recommendations: Dict) -> str:
    """Generate specialized analysis for reader/writer configuration"""
    analysis = "Reader/Writer Configuration Analysis:\n\n"
    
    for env, config in recommendations.items():
        writer = config['writer_instance_class']
        readers = config['reader_count']
        reader_type = config['reader_instance_class']
        env_type = config['environment_type']
        
        analysis += f"- {env} ({env_type.title()}):\n"
        analysis += f"  Writer: {writer}\n"
        analysis += f"  Readers: {readers} x {reader_type}\n"
        
        # Add AI assessment
        if env_type == 'production':
            if readers == 0 and config['read_write_ratio'] > 50:
                analysis += "  ⚠️ Recommendation: Add read replicas for better read scaling\n"
            elif readers > 0 and config['read_write_ratio'] < 30:
                analysis += "  ℹ️ Suggestion: Consider reducing read replicas to optimize costs\n"
            else:
                analysis += "  ✅ Configuration appears well-balanced\n"
        else:
            if readers > 0:
                analysis += "  💡 Note: Non-production environments typically don't need read replicas\n"
    
    return analysis
     
        
    def calculate_migration_costs(self, recommendations: Dict, migration_params: Dict) -> Dict:
        """Calculate migration costs based on recommendations"""
        
        region = migration_params.get('region', 'us-east-1')
        target_engine = migration_params.get('target_engine', 'postgres')
        
        total_monthly_cost = 0
        environment_costs = {}
        
        for env_name, rec in recommendations.items():
            env_costs = self._calculate_environment_cost(env_name, rec, region, target_engine)
            environment_costs[env_name] = env_costs
            total_monthly_cost += env_costs['total_monthly']
        
        # Migration service costs
        data_size_gb = migration_params.get('data_size_gb', 1000)
        migration_timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        # DMS costs
        dms_instance_cost = 0.2 * 24 * 7 * migration_timeline_weeks  # t3.large instance
        
        # Data transfer costs
        transfer_costs = self._calculate_transfer_costs(data_size_gb, migration_params)
        
        # Professional services
        ps_cost = migration_timeline_weeks * 8000  # $8k per week
        
        migration_costs = {
            'dms_instance': dms_instance_cost,
            'data_transfer': transfer_costs.get('total', data_size_gb * 0.09),
            'professional_services': ps_cost,
            'contingency': 0,
            'total': 0
        }
        # Calculate contingency and total
        base_cost = migration_costs['dms_instance'] + migration_costs['data_transfer'] + migration_costs['professional_services']
        migration_costs['contingency'] = base_cost * 0.2
        migration_costs['total'] = base_cost + migration_costs['contingency']
        
        return {
            'monthly_aws_cost': total_monthly_cost,
            'annual_aws_cost': total_monthly_cost * 12,
            'environment_costs': environment_costs,
            'migration_costs': migration_costs,
            'transfer_costs': transfer_costs
        }
    
def generate_ai_insights_sync(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
    """Generate REAL Claude AI insights synchronously"""
    
    if not self.anthropic_api_key:
        return {'error': 'No Anthropic API key provided', 'source': 'Error'}
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        
        context = f"""
        You are an AWS database migration expert. Analyze this project:
        
        MIGRATION: {migration_params.get('source_engine')} → {migration_params.get('target_engine')}
        DATA SIZE: {migration_params.get('data_size_gb', 0):,} GB
        TIMELINE: {migration_params.get('migration_timeline_weeks', 0)} weeks
        MONTHLY COST: ${cost_analysis.get('monthly_aws_cost', 0):,.0f}
        MIGRATION COST: ${cost_analysis.get('migration_costs', {}).get('total', 0):,.0f}
        
        Provide specific insights for:
        1. Top 3 migration risks and how to mitigate them
        2. Cost optimization opportunities
        3. Timeline feasibility and recommendations
        4. Technical considerations for this specific database migration
        
        Be concise but actionable.
        """
        
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            temperature=0.3,
            messages=[{"role": "user", "content": context}]
        )
        
        return {
            'ai_analysis': message.content[0].text,
            'source': 'Claude AI (Real)',
            'model': 'claude-3-sonnet-20240229',
            'success': True
        }
        
    except ImportError:
        return {'error': 'Run: pip install anthropic', 'source': 'Library Error', 'success': False}
    except Exception as e:
        return {'error': f'Claude AI failed: {str(e)}', 'source': 'API Error', 'success': False}
    
 
# FIXED: Synchronous Claude AI Implementation for Streamlit
class StreamlitClaudeAIAnalyzer:
    """Synchronous Claude AI integration that works with Streamlit"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        
        if api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Anthropic client: {e}")
    
    def generate_ai_insights_sync(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Generate AI insights synchronously for Streamlit"""
        
        if not self.client:
            return {
                'error': 'Claude AI not available - using fallback insights',
                'fallback_insights': self._get_fallback_insights(cost_analysis, migration_params)
            }
        
        try:
            # Prepare context for Claude
            context = self._prepare_migration_context(cost_analysis, migration_params)
            
            # Create the prompt
            prompt = f"""
            You are an AWS migration expert analyzing a database migration project. 
            Please provide detailed insights and recommendations based on the following migration analysis:

            {context}

            Please provide insights in the following categories:
            1. Cost Analysis & Optimization
            2. Risk Assessment
            3. Migration Strategy Recommendations
            4. Timeline Feasibility
            5. Technical Considerations
            6. Post-Migration Optimization

            Format your response as a structured analysis with specific, actionable recommendations.
            Be concise but comprehensive. Limit response to 1500 words.
            """
            
            # Call Claude API synchronously
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse Claude's response
            ai_response = message.content[0].text
            
            # Structure the response
            structured_insights = self._structure_ai_response(ai_response)
            structured_insights['source'] = 'Claude AI'
            structured_insights['model'] = 'claude-3-sonnet-20240229'
            structured_insights['api_success'] = True
            
            return structured_insights
            
        except Exception as e:
            print(f"Error calling Claude AI: {e}")
            return {
                'error': f'Claude AI call failed: {str(e)}',
                'fallback_insights': self._get_fallback_insights(cost_analysis, migration_params),
                'api_success': False
            }
    
    def _prepare_migration_context(self, cost_analysis: Dict, migration_params: Dict) -> str:
        """Prepare context for Claude AI"""
        
        # Calculate some additional metrics
        total_environments = len(cost_analysis.get('environment_costs', {}))
        avg_monthly_cost = cost_analysis.get('monthly_aws_cost', 0) / max(total_environments, 1)
        
        context = f"""
        MIGRATION PROJECT OVERVIEW:
        - Source Database: {migration_params.get('source_engine', 'Unknown')}
        - Target Database: {migration_params.get('target_engine', 'Unknown')}
        - Data Size: {migration_params.get('data_size_gb', 0):,} GB
        - Timeline: {migration_params.get('migration_timeline_weeks', 0)} weeks
        - Team Size: {migration_params.get('team_size', 0)} members
        - Team Expertise: {migration_params.get('team_expertise', 'Unknown')}
        - Budget: ${migration_params.get('migration_budget', 0):,}

        COST ANALYSIS:
        - Monthly AWS Cost: ${cost_analysis.get('monthly_aws_cost', 0):,.0f}
        - Annual AWS Cost: ${cost_analysis.get('annual_aws_cost', 0):,.0f}
        - Migration Investment: ${cost_analysis.get('migration_costs', {}).get('total', 0):,.0f}
        - Average Cost per Environment: ${avg_monthly_cost:,.0f}/month

        ENVIRONMENTS:
        - Number of Environments: {total_environments}
        - Environment Names: {', '.join(cost_analysis.get('environment_costs', {}).keys())}

        MIGRATION COMPLEXITY FACTORS:
        - Applications Connected: {migration_params.get('num_applications', 0)}
        - Stored Procedures/Functions: {migration_params.get('num_stored_procedures', 0)}
        - Region: {migration_params.get('region', 'us-east-1')}
        - Direct Connect Available: {migration_params.get('use_direct_connect', False)}
        
        COST BREAKDOWN:
        - DMS Instance Cost: ${cost_analysis.get('migration_costs', {}).get('dms_instance', 0):,.0f}
        - Data Transfer Cost: ${cost_analysis.get('migration_costs', {}).get('data_transfer', 0):,.0f}
        - Professional Services: ${cost_analysis.get('migration_costs', {}).get('professional_services', 0):,.0f}
        """
        
        return context
    
    def _structure_ai_response(self, ai_response: str) -> Dict:
        """Structure Claude's response into categories"""
        
        # Simple parsing to extract key sections
        sections = ai_response.split('\n\n')
        
        structured = {
            'cost_optimization': '',
            'risk_mitigation': '',
            'migration_strategy': '',
            'timeline_recommendations': '',
            'technical_considerations': '',
            'post_migration_optimization': '',
            'key_recommendations': [],
            'executive_summary': '',
            'full_response': ai_response
        }
        
        # Extract key recommendations (look for numbered lists or bullet points)
        recommendations = []
        for section in sections:
            lines = section.split('\n')
            for line in lines:
                line = line.strip()
                if (line.startswith('•') or line.startswith('-') or 
                    any(line.startswith(f"{i}.") for i in range(1, 10))):
                    recommendations.append(line.lstrip('•-0123456789. '))
        
        structured['key_recommendations'] = recommendations[:8]  # Top 8 recommendations
        
        # Try to extract sections based on keywords
        current_section = ''
        for section in sections:
            section_lower = section.lower()
            
            if any(keyword in section_lower for keyword in ['cost', 'pricing', 'financial', 'budget']):
                structured['cost_optimization'] += section + '\n\n'
            elif any(keyword in section_lower for keyword in ['risk', 'mitigation', 'challenges']):
                structured['risk_mitigation'] += section + '\n\n'
            elif any(keyword in section_lower for keyword in ['strategy', 'approach', 'methodology']):
                structured['migration_strategy'] += section + '\n\n'
            elif any(keyword in section_lower for keyword in ['timeline', 'schedule', 'phases']):
                structured['timeline_recommendations'] += section + '\n\n'
            elif any(keyword in section_lower for keyword in ['technical', 'architecture', 'configuration']):
                structured['technical_considerations'] += section + '\n\n'
            elif any(keyword in section_lower for keyword in ['optimization', 'performance', 'tuning']):
                structured['post_migration_optimization'] += section + '\n\n'
            elif any(keyword in section_lower for keyword in ['summary', 'conclusion', 'overview']):
                structured['executive_summary'] += section + '\n\n'
        
        # If executive summary is empty, create one from the first section
        if not structured['executive_summary'] and sections:
            structured['executive_summary'] = sections[0]
        
        return structured
    
    def _get_fallback_insights(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Enhanced fallback insights when AI is unavailable"""
        
        monthly_cost = cost_analysis.get('monthly_aws_cost', 0)
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        data_size = migration_params.get('data_size_gb', 0)
        team_size = migration_params.get('team_size', 5)
        source_engine = migration_params.get('source_engine', 'unknown')
        target_engine = migration_params.get('target_engine', 'unknown')
        
        # Generate more intelligent fallback based on data
        risk_level = "Medium"
        if timeline_weeks < 8 or data_size > 10000 or team_size < 3:
            risk_level = "High"
        elif timeline_weeks > 16 and data_size < 1000 and team_size >= 5:
            risk_level = "Low"
        
        cost_efficiency = "Good"
        if monthly_cost > 10000:
            cost_efficiency = "Review required - high monthly cost"
        elif monthly_cost < 1000:
            cost_efficiency = "Very efficient"
        
        return {
            'cost_optimization': f"Monthly AWS cost of ${monthly_cost:,.0f} appears {cost_efficiency.lower()} for this migration scale. Consider Reserved Instances for 30-40% savings on production workloads. Evaluate right-sizing opportunities post-migration.",
            
            'risk_mitigation': f"Migration risk level: {risk_level}. Key risks include {source_engine} to {target_engine} compatibility, application dependencies, and data validation. Implement comprehensive testing strategy and maintain rollback procedures.",
            
            'migration_strategy': f"Recommended approach: Phased migration starting with non-production environments. Use AWS DMS for continuous replication with minimal downtime cutover. Timeline of {timeline_weeks} weeks allows for proper validation.",
            
            'timeline_recommendations': f"Timeline allocation: 20% planning, 40% migration execution, 30% testing/validation, 10% go-live. With {data_size:,} GB of data, allocate sufficient time for initial load and ongoing synchronization.",
            
            'technical_considerations': f"Technical focus areas: Schema conversion ({source_engine} → {target_engine}), stored procedure migration, index optimization, and connection string updates across {migration_params.get('num_applications', 'unknown')} applications.",
            
            'post_migration_optimization': "Post-migration opportunities: Instance right-sizing based on actual usage, storage optimization, backup strategy refinement, and performance monitoring implementation. Plan optimization review 30-60 days post-migration.",
            
            'key_recommendations': [
                "Start with development/QA environments for validation",
                "Implement comprehensive backup and rollback procedures",
                "Use AWS DMS for minimal downtime migration",
                "Plan for application connection string updates",
                "Establish performance baselines before migration",
                "Consider Reserved Instances for cost optimization",
                "Implement monitoring and alerting for new environment",
                "Schedule post-migration optimization review"
            ],
            
            'executive_summary': f"Migration from {source_engine} to {target_engine} is feasible within {timeline_weeks} weeks with {risk_level.lower()} risk level. Monthly operational cost of ${monthly_cost:,.0f} provides good value. Success depends on proper planning, testing, and team preparation.",
            
            'source': 'Enhanced Fallback Analysis'
        }


# FIXED: Synchronous Migration Analyzer for Streamlit
class StreamlitMigrationAnalyzer:
    """Migration analyzer that works synchronously with Streamlit"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.pricing_api = RealAWSPricingAPI()
        self.ai_analyzer = StreamlitClaudeAIAnalyzer(anthropic_api_key)
        self.anthropic_api_key = anthropic_api_key
    
    def calculate_instance_recommendations(self, environment_specs: Dict) -> Dict:
        """Calculate AWS instance recommendations using real pricing"""
        
        recommendations = {}
        
        for env_name, specs in environment_specs.items():
            cpu_cores = specs['cpu_cores']
            ram_gb = specs['ram_gb']
            storage_gb = specs['storage_gb']
            
            # Determine environment type
            environment_type = self._categorize_environment(env_name)
            
            # Calculate instance class
            instance_class = self._calculate_instance_class(cpu_cores, ram_gb, environment_type)
            
            # Multi-AZ recommendation
            multi_az = environment_type in ['production', 'staging']
            
            recommendations[env_name] = {
                'environment_type': environment_type,
                'instance_class': instance_class,
                'cpu_cores': cpu_cores,
                'ram_gb': ram_gb,
                'storage_gb': storage_gb,
                'multi_az': multi_az,
                'daily_usage_hours': specs.get('daily_usage_hours', 24),
                'peak_connections': specs.get('peak_connections', 100)
            }
        
        return recommendations
    
    def calculate_migration_costs(self, recommendations: Dict, migration_params: Dict) -> Dict:
        """Calculate migration costs using real AWS pricing"""
        
        region = migration_params.get('region', 'us-east-1')
        target_engine = migration_params.get('target_engine', 'postgres')
        
        total_monthly_cost = 0
        environment_costs = {}
        
        for env_name, rec in recommendations.items():
            # Use real pricing API
            env_costs = self._calculate_environment_cost_real(env_name, rec, region, target_engine)
            environment_costs[env_name] = env_costs
            total_monthly_cost += env_costs['total_monthly']
        
        # Migration service costs
        data_size_gb = migration_params.get('data_size_gb', 1000)
        migration_timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        # DMS costs
        dms_instance_cost = 0.2 * 24 * 7 * migration_timeline_weeks
        
        # Data transfer costs
        transfer_costs = self._calculate_transfer_costs(data_size_gb, migration_params)
        
        # Professional services
        ps_cost = migration_timeline_weeks * 8000
        
        migration_costs = {
            'dms_instance': dms_instance_cost,
            'data_transfer': transfer_costs.get('total', data_size_gb * 0.09),
            'professional_services': ps_cost,
            'contingency': 0,
            'total': 0
        }
        
        base_cost = migration_costs['dms_instance'] + migration_costs['data_transfer'] + migration_costs['professional_services']
        migration_costs['contingency'] = base_cost * 0.2
        migration_costs['total'] = base_cost + migration_costs['contingency']
        
        return {
            'monthly_aws_cost': total_monthly_cost,
            'annual_aws_cost': total_monthly_cost * 12,
            'environment_costs': environment_costs,
            'migration_costs': migration_costs,
            'transfer_costs': transfer_costs
        }
    
    def generate_ai_insights_sync(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Generate AI insights synchronously for Streamlit"""
        return self.ai_analyzer.generate_ai_insights_sync(cost_analysis, migration_params)
    
    def _calculate_environment_cost_real(self, env_name: str, rec: Dict, region: str, target_engine: str) -> Dict:
        """Calculate environment cost using real AWS pricing"""
        
        # Get real pricing from AWS API
        pricing = self.pricing_api.get_rds_pricing(
            region, target_engine, rec['instance_class'], rec['multi_az']
        )
        
        # Calculate monthly hours
        daily_hours = rec['daily_usage_hours']
        monthly_hours = daily_hours * 30
        
        # Instance cost using real pricing
        instance_cost = pricing['hourly'] * monthly_hours
        
        # Storage cost using real pricing
        storage_cost = rec['storage_gb'] * pricing['storage_gb']
        
        # Backup cost (estimate 20% of storage)
        backup_cost = storage_cost * 0.2
        
        # Total monthly cost
        total_monthly = instance_cost + storage_cost + backup_cost
        
        return {
            'instance_cost': instance_cost,
            'storage_cost': storage_cost,
            'backup_cost': backup_cost,
            'total_monthly': total_monthly,
            'pricing_source': pricing.get('source', 'Unknown')
        }
    
    def _categorize_environment(self, env_name: str) -> str:
        """Categorize environment type from name"""
        env_lower = env_name.lower()
        if any(term in env_lower for term in ['prod', 'production', 'prd']):
            return 'production'
        elif any(term in env_lower for term in ['stag', 'staging', 'preprod']):
            return 'staging'
        elif any(term in env_lower for term in ['qa', 'test', 'uat', 'sqa']):
            return 'testing'
        elif any(term in env_lower for term in ['dev', 'development', 'sandbox']):
            return 'development'
        return 'production'
    
    def _calculate_instance_class(self, cpu_cores: int, ram_gb: int, env_type: str) -> str:
        """Calculate appropriate instance class"""
        
        if cpu_cores <= 2 and ram_gb <= 8:
            instance_class = 'db.t3.medium'
        elif cpu_cores <= 4 and ram_gb <= 16:
            instance_class = 'db.t3.large'
        elif cpu_cores <= 8 and ram_gb <= 32:
            instance_class = 'db.r5.large'
        elif cpu_cores <= 16 and ram_gb <= 64:
            instance_class = 'db.r5.xlarge'
        elif cpu_cores <= 32 and ram_gb <= 128:
            instance_class = 'db.r5.2xlarge'
        elif cpu_cores <= 64 and ram_gb <= 256:
            instance_class = 'db.r5.4xlarge'
        else:
            instance_class = 'db.r5.8xlarge'
        
        # Environment-specific adjustments
        if env_type == 'development' and 'r5' in instance_class:
            downsized = {
                'db.r5.8xlarge': 'db.r5.4xlarge',
                'db.r5.4xlarge': 'db.r5.2xlarge',
                'db.r5.2xlarge': 'db.r5.xlarge',
                'db.r5.xlarge': 'db.r5.large',
                'db.r5.large': 'db.t3.large'
            }
            instance_class = downsized.get(instance_class, instance_class)
        
        return instance_class
    
    def _calculate_transfer_costs(self, data_size_gb: int, migration_params: Dict) -> Dict:
        """Calculate data transfer costs"""
        
        use_direct_connect = migration_params.get('use_direct_connect', False)
        
        internet_cost = data_size_gb * 0.09
        
        if use_direct_connect:
            dx_cost = data_size_gb * 0.02
        else:
            dx_cost = internet_cost
        
        return {
            'internet': internet_cost,
            'direct_connect': dx_cost,
            'total': min(internet_cost, dx_cost)
        }


# FIXED: Streamlit-compatible analysis function
def run_streamlit_migration_analysis():
    """Run migration analysis synchronously for Streamlit"""
    
    try:
        
         # Check if reconciliation is enabled
        reconciliation_settings = getattr(st.session_state, 'reconciliation_settings', {})
        use_reconciliation = reconciliation_settings.get('auto_reconcile', False)
        
        # Initialize analyzer with reconciliation if enabled
        anthropic_api_key = st.session_state.migration_params.get('anthropic_api_key')
        
        if use_reconciliation:
            st.info("🔄 Using reconciliation-aware pricing API")
            pricing_api = ReconciliationAWSPricingAPI()
            analyzer = MigrationAnalyzer(anthropic_api_key)
            analyzer.pricing_api = pricing_api
        else:
            analyzer = MigrationAnalyzer(anthropic_api_key)
        
        # Check if this is enhanced environment data
        is_enhanced = is_enhanced_environment_data(st.session_state.environment_specs)
        
        if is_enhanced:
            # Run enhanced analysis (if available)
            st.info("🔬 Detected enhanced environment configuration - running cluster analysis")
            run_enhanced_migration_analysis()
        else:
            # Run streamlit-compatible analysis
            st.info("📊 Running real-time migration analysis")
            
            # Initialize STREAMLIT-compatible analyzer
            anthropic_api_key = st.session_state.migration_params.get('anthropic_api_key')
            analyzer = StreamlitMigrationAnalyzer(anthropic_api_key)
            
            # Step 1: Calculate recommendations with real AWS pricing
            st.write("📊 Calculating instance recommendations with live AWS pricing...")
            recommendations = analyzer.calculate_instance_recommendations(st.session_state.environment_specs)
            st.session_state.recommendations = recommendations
            
            # Step 2: Calculate costs with real AWS pricing
            st.write("💰 Analyzing costs with real-time AWS pricing...")
            cost_analysis = analyzer.calculate_migration_costs(recommendations, st.session_state.migration_params)
            
            # Check pricing sources
            pricing_sources = set()
            for env_costs in cost_analysis['environment_costs'].values():
                pricing_sources.add(env_costs.get('pricing_source', 'Unknown'))
            
            if 'AWS Pricing API' in pricing_sources:
                st.success("✅ Using real-time AWS pricing data")
            else:
                st.warning("⚠️ Using fallback pricing data (AWS API unavailable)")
            
            st.session_state.analysis_results = cost_analysis
            
            # Step 3: Risk assessment
            st.write("⚠️ Assessing risks...")
            risk_assessment = create_default_risk_assessment()
            st.session_state.risk_assessment = risk_assessment
            
            # Step 4: AI insights (SYNCHRONOUS)
            if anthropic_api_key:
                st.write("🤖 Generating AI insights with Claude...")
                try:
                    # Call synchronously - NO AWAIT!
                    ai_insights = analyzer.generate_ai_insights_sync(cost_analysis, st.session_state.migration_params)
                    st.session_state.ai_insights = ai_insights
                    
                    if ai_insights.get('source') == 'Claude AI':
                        st.success("✅ AI insights generated by Claude AI")
                    else:
                        st.warning("⚠️ Using enhanced fallback insights (Claude API unavailable)")
                        
                except Exception as e:
                    st.warning(f"Claude AI call failed: {str(e)}")
                    st.session_state.ai_insights = {'error': str(e)}
            else:
                st.info("ℹ️ Provide Anthropic API key for Claude AI insights")
            
            st.success("✅ Analysis complete with real-time data!")
        
        # Show quick summary
        show_analysis_summary()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        # Ensure we have fallback data
        if not hasattr(st.session_state, 'risk_assessment') or st.session_state.risk_assessment is None:
            st.session_state.risk_assessment = get_fallback_risk_assessment()

def show_analysis_summary():
    """Show analysis summary after completion"""
    
    st.markdown("#### 🎯 Analysis Summary")
    col1, col2, col3 = st.columns(3)
    
    # Get results from whichever analysis was run
    if hasattr(st.session_state, 'enhanced_analysis_results') and st.session_state.enhanced_analysis_results:
        results = st.session_state.enhanced_analysis_results
    else:
        results = st.session_state.analysis_results
    
    if results:
        with col1:
            st.metric("Monthly Cost", f"${results['monthly_aws_cost']:,.0f}")
        
        with col2:
            migration_cost = results.get('migration_costs', {}).get('total', 0)
            st.metric("Migration Cost", f"${migration_cost:,.0f}")
        
        with col3:
            if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
                risk_level = st.session_state.risk_assessment['risk_level']['level']
                st.metric("Risk Level", risk_level)
            else:
                st.metric("Risk Level", "Medium")
    
    # Provide navigation hint
    st.info("📈 View detailed results in the 'Results Dashboard' section")


# FIXED: Update the analysis section to use synchronous function
def show_analysis_section_fixed():
    """Show analysis and recommendations section - FIXED for Streamlit"""
    
    st.markdown("## 🚀 Migration Analysis & Recommendations")
    
    # Check prerequisites
    if not st.session_state.migration_params:
        st.error("❌ Migration configuration required")
        st.info("👆 Please complete the 'Migration Configuration' section first")
        return
    
    if not st.session_state.environment_specs:
        st.error("❌ Environment configuration required")
        st.info("👆 Please complete the 'Environment Setup' section first")
        return
    
    # Display current configuration
    st.markdown("### 📋 Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        params = st.session_state.migration_params
        st.markdown(f"""
        **Migration Type:** {params['source_engine']} → {params['target_engine']}  
        **Data Size:** {params['data_size_gb']:,} GB  
        **Timeline:** {params['migration_timeline_weeks']} weeks  
        **Team Size:** {params['team_size']} members  
        **Budget:** ${params['migration_budget']:,}
        """)
    
    with col2:
        envs = st.session_state.environment_specs
        st.markdown(f"**Environments:** {len(envs)}")
        
        # Show first few environments
        count = 0
        for env_name, specs in envs.items():
            if count < 4:
                cpu_cores = specs.get('cpu_cores', 'N/A')
                ram_gb = specs.get('ram_gb', 'N/A')
                st.markdown(f"• **{env_name}:** {cpu_cores} cores, {ram_gb} GB RAM")
                count += 1
        
        if len(envs) > 4:
            st.markdown(f"• ... and {len(envs) - 4} more environments")
    
    # API status check
    show_api_status_inline()
    
    # Run analysis - FIXED VERSION
    if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
        # Clear any previous results
        st.session_state.analysis_results = None
        if hasattr(st.session_state, 'enhanced_analysis_results'):
            st.session_state.enhanced_analysis_results = None
        
        with st.spinner("🔄 Analyzing migration requirements with real-time data..."):
            run_streamlit_migration_analysis()  # Use the FIXED synchronous function


def show_api_status_inline():
    """Show inline API status"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Check AWS status
        try:
            import boto3
            boto3.client('pricing', region_name='us-east-1')
            st.success("✅ AWS Pricing API: Ready")
        except ImportError:
            st.error("❌ AWS: Missing boto3 library")
        except Exception:
            st.warning("⚠️ AWS: Check credentials")
    
    with col2:
        # Check Claude status
        anthropic_key = st.session_state.migration_params.get('anthropic_api_key')
        
        if anthropic_key:
            try:
                import anthropic
                st.success("✅ Claude AI: Ready")
            except ImportError:
                st.error("❌ Claude: Missing anthropic library")
        else:
            st.info("ℹ️ Claude AI: No API key provided")

class MigrationAnalyzer:
    """Basic migration analyzer for standard environment configurations"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.pricing_api = EnhancedAWSPricingAPI()
        self.anthropic_api_key = anthropic_api_key
    
    def calculate_instance_recommendations(self, environment_specs: Dict) -> Dict:
        """Calculate AWS instance recommendations for environments"""
        
        recommendations = {}
        
        for env_name, specs in environment_specs.items():
            cpu_cores = specs['cpu_cores']
            ram_gb = specs['ram_gb']
            storage_gb = specs['storage_gb']
            
            # Determine environment type
            environment_type = self._categorize_environment(env_name)
            
            # Calculate instance class
            instance_class = self._calculate_instance_class(cpu_cores, ram_gb, environment_type)
            
            # Multi-AZ recommendation
            multi_az = environment_type in ['production', 'staging']
                                  
            recommendations[env_name] = {
                'environment_type': environment_type,
                'instance_class': instance_class,
                'cpu_cores': cpu_cores,
                'ram_gb': ram_gb,
                'storage_gb': storage_gb,
                'multi_az': multi_az,
                'daily_usage_hours': specs.get('daily_usage_hours', 24),
                'peak_connections': specs.get('peak_connections', 100)
            }
        
        
        
       
        return recommendations
    
    def calculate_migration_costs(self, recommendations: Dict, migration_params: Dict) -> Dict:
        """Calculate migration costs based on recommendations"""
        
        region = migration_params.get('region', 'us-east-1')
        target_engine = migration_params.get('target_engine', 'postgres')
        
        total_monthly_cost = 0
        environment_costs = {}
        
        for env_name, rec in recommendations.items():
            env_costs = self._calculate_environment_cost(env_name, rec, region, target_engine)
            environment_costs[env_name] = env_costs
            total_monthly_cost += env_costs['total_monthly']
               
        
        # Migration service costs
        data_size_gb = migration_params.get('data_size_gb', 1000)
        migration_timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        # DMS costs
        dms_instance_cost = 0.2 * 24 * 7 * migration_timeline_weeks  # t3.large instance
        
        # Data transfer costs
        transfer_costs = self._calculate_transfer_costs(data_size_gb, migration_params)
        
        # Professional services
        ps_cost = migration_timeline_weeks * 8000  # $8k per week
        
        migration_costs = {
            'dms_instance': dms_instance_cost,
            'data_transfer': transfer_costs.get('total', data_size_gb * 0.09),
            'professional_services': ps_cost,
            'contingency': 0,
            'total': 0
        }
        # Calculate contingency and total
        base_cost = migration_costs['dms_instance'] + migration_costs['data_transfer'] + migration_costs['professional_services']
        migration_costs['contingency'] = base_cost * 0.2
        migration_costs['total'] = base_cost + migration_costs['contingency']
        
        return {
            'monthly_aws_cost': total_monthly_cost,
            'annual_aws_cost': total_monthly_cost * 12,
            'environment_costs': environment_costs,
            'migration_costs': migration_costs,
            'transfer_costs': transfer_costs
        }
    
    async def generate_ai_insights(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Generate AI insights (optional, requires API key)"""
        
        if not self.anthropic_api_key:
            return {'error': 'No API key provided'}
        
        try:
            # Simple insights without actual API call for now
            insights = {
                'cost_efficiency': f"Monthly AWS cost of ${cost_analysis['monthly_aws_cost']:,.0f} represents efficient cloud migration",
                'migration_strategy': f"Recommended {migration_params.get('migration_timeline_weeks', 12)}-week timeline is appropriate for this scale",
                'risk_assessment': "Migration complexity is manageable with proper planning and resources"
            }
            return insights
        except Exception as e:
            return {'error': str(e)}
    
    def _categorize_environment(self, env_name: str) -> str:
        """Categorize environment type from name"""
        env_lower = env_name.lower()
        if any(term in env_lower for term in ['prod', 'production', 'prd']):
            return 'production'
        elif any(term in env_lower for term in ['stag', 'staging', 'preprod']):
            return 'staging'
        elif any(term in env_lower for term in ['qa', 'test', 'uat', 'sqa']):
            return 'testing'
        elif any(term in env_lower for term in ['dev', 'development', 'sandbox']):
            return 'development'
        return 'production'  # Default to production for safety
    
    def _calculate_instance_class(self, cpu_cores: int, ram_gb: int, env_type: str) -> str:
        """Calculate appropriate instance class"""
        
        # Instance sizing logic
        if cpu_cores <= 2 and ram_gb <= 8:
            instance_class = 'db.t3.medium'
        elif cpu_cores <= 4 and ram_gb <= 16:
            instance_class = 'db.t3.large'
        elif cpu_cores <= 8 and ram_gb <= 32:
            instance_class = 'db.r5.large'
        elif cpu_cores <= 16 and ram_gb <= 64:
            instance_class = 'db.r5.xlarge'
        elif cpu_cores <= 32 and ram_gb <= 128:
            instance_class = 'db.r5.2xlarge'
        elif cpu_cores <= 64 and ram_gb <= 256:
            instance_class = 'db.r5.4xlarge'
        else:
            instance_class = 'db.r5.8xlarge'
        
        # Environment-specific adjustments
        if env_type == 'development' and 'r5' in instance_class:
            # Downsize for development
            downsized = {
                'db.r5.8xlarge': 'db.r5.4xlarge',
                'db.r5.4xlarge': 'db.r5.2xlarge',
                'db.r5.2xlarge': 'db.r5.xlarge',
                'db.r5.xlarge': 'db.r5.large',
                'db.r5.large': 'db.t3.large'
            }
            instance_class = downsized.get(instance_class, instance_class)
        
        return instance_class
    
    def _calculate_environment_cost(self, env_name: str, rec: Dict, region: str, target_engine: str) -> Dict:
        """Calculate cost for a single environment"""
        
        # Get pricing
        pricing = self.pricing_api.get_rds_pricing(
            region, target_engine, rec['instance_class'], rec['multi_az']
        )
        
        # Calculate monthly hours
        daily_hours = rec['daily_usage_hours']
        monthly_hours = daily_hours * 30
        
        # Instance cost
        instance_cost = pricing['hourly'] * monthly_hours
        
        # Storage cost
        storage_cost = rec['storage_gb'] * pricing['storage_gb']
        
        # Backup cost (estimate 20% of storage)
        backup_cost = storage_cost * 0.2
        
        # Total monthly cost
        total_monthly = instance_cost + storage_cost + backup_cost
        
        return {
            'instance_cost': instance_cost,
            'storage_cost': storage_cost,
            'backup_cost': backup_cost,
            'total_monthly': total_monthly
        }
    
    def _calculate_transfer_costs(self, data_size_gb: int, migration_params: Dict) -> Dict:
        """Calculate data transfer costs"""
        
        use_direct_connect = migration_params.get('use_direct_connect', False)
        
        # Internet transfer
        internet_cost = data_size_gb * 0.09  # $0.09 per GB
        
        # Direct Connect transfer
        if use_direct_connect:
            dx_cost = data_size_gb * 0.02  # $0.02 per GB
        else:
            dx_cost = internet_cost
        
        return {
            'internet': internet_cost,
            'direct_connect': dx_cost,
            'total': min(internet_cost, dx_cost)
        }
        
class ImprovedReportGenerator:
    """Enhanced PDF Report Generator with Better Formatting and Layout"""
    
    def __init__(self):
        """Initialize the improved report generator"""
        self.styles = getSampleStyleSheet()
        self.chart_width = 5.5*inch
        self.chart_height = 3.5*inch
        self.setup_custom_styles()
        
        # Debug: Verify styles are created
        self._verify_styles()
    
    def _verify_styles(self):
        """Verify that all required styles are available"""
        required_styles = [
            'ReportTitle', 'SectionHeader', 'SubsectionHeader', 
            'BodyText', 'KeyMetric', 'TableHeader', 'BulletPoint'
        ]
        
        for style_name in required_styles:
            if style_name not in self.styles.byName:
                print(f"Warning: Style '{style_name}' not found. Creating fallback.")
                self._create_fallback_style(style_name)
    
    def _create_fallback_style(self, style_name):
        """Create a fallback style if the original creation failed"""
        try:
            base_style = self.styles['Normal']
            if style_name == 'ReportTitle':
                self.styles.add(ParagraphStyle(
                    name='ReportTitle',
                    parent=self.styles['Title'],
                    fontSize=24,
                    fontName='Helvetica-Bold',
                    alignment=TA_CENTER
                ))
            elif style_name == 'SectionHeader':
                self.styles.add(ParagraphStyle(
                    name='SectionHeader',
                    parent=self.styles['Heading1'],
                    fontSize=16,
                    fontName='Helvetica-Bold',
                    spaceBefore=20,
                    spaceAfter=12
                ))
            else:
                # Generic fallback
                self.styles.add(ParagraphStyle(
                    name=style_name,
                    parent=base_style,
                    fontSize=12,
                    fontName='Helvetica'
                ))
        except Exception as e:
            print(f"Error creating fallback style {style_name}: {e}")
    
    def setup_custom_styles(self):
        """Setup improved custom styles for the report"""
        try:
            # Report Title Style
            self.styles.add(ParagraphStyle(
                name='ReportTitle',
                parent=self.styles['Title'],
                fontSize=24,
                spaceBefore=0,
                spaceAfter=20,
                textColor=colors.darkblue,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
            
            # Section Header Style
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading1'],
                fontSize=16,
                spaceBefore=20,
                spaceAfter=12,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold',
                borderWidth=1,
                borderColor=colors.darkblue,
                borderPadding=5
            ))
            
            # Subsection Header Style
            self.styles.add(ParagraphStyle(
                name='SubsectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceBefore=15,
                spaceAfter=8,
                textColor=colors.darkgreen,
                fontName='Helvetica-Bold'
            ))
            
            # Body Text Style
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceBefore=6,
                spaceAfter=6,
                textColor=colors.black,
                fontName='Helvetica',
                leading=14
            ))
            
            # Key Metric Style
            self.styles.add(ParagraphStyle(
                name='KeyMetric',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.darkred,
                fontName='Helvetica-Bold',
                alignment=TA_CENTER
            ))
            
            # Table Header Style
            self.styles.add(ParagraphStyle(
                name='TableHeader',
                parent=self.styles['Normal'],
                fontSize=10,
                textColor=colors.white,
                fontName='Helvetica-Bold',
                alignment=TA_CENTER
            ))
            
            # Bullet Point Style
            self.styles.add(ParagraphStyle(
                name='BulletPoint',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceBefore=4,
                spaceAfter=4,
                leftIndent=20,
                bulletIndent=10,
                fontName='Helvetica'
            ))
            
        except Exception as e:
            print(f"Error setting up custom styles: {e}")
            self._create_basic_styles()
    
    def _create_basic_styles(self):
        """Create basic styles as fallback"""
        basic_styles = {
            'ReportTitle': ('Title', 24),
            'SectionHeader': ('Heading1', 16),
            'SubsectionHeader': ('Heading2', 14),
            'BodyText': ('Normal', 11),
            'KeyMetric': ('Normal', 12),
            'TableHeader': ('Normal', 10),
            'BulletPoint': ('Normal', 11)
        }
        
        for style_name, (parent, size) in basic_styles.items():
            try:
                if style_name not in self.styles.byName:
                    self.styles.add(ParagraphStyle(
                        name=style_name,
                        parent=self.styles[parent],
                        fontSize=size,
                        fontName='Helvetica'
                    ))
            except Exception as e:
                print(f"Error creating basic style {style_name}: {e}")
    
    def safe_get_style(self, style_name, fallback='Normal'):
        """Safely get a style, falling back to Normal if not found"""
        try:
            return self.styles[style_name]
        except KeyError:
            print(f"Style '{style_name}' not found, using '{fallback}'")
            return self.styles[fallback]
    
    def create_improved_cost_chart(self, analysis_results, analysis_mode):
        """Create an improved cost breakdown chart with better formatting"""
        try:
            if analysis_mode == 'single':
                valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
                if not valid_results:
                    return None
                
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                
                # Extract cost components with better logic
                cost_data = {}
                
                if 'writer' in prod_result:
                    # Aurora/RDS Cluster setup
                    writer_cost = prod_result.get('cost_breakdown', {}).get('writer_monthly', 0)
                    readers_cost = prod_result.get('cost_breakdown', {}).get('readers_monthly', 0)
                    storage_cost = prod_result.get('cost_breakdown', {}).get('storage_monthly', 0)
                    backup_cost = prod_result.get('cost_breakdown', {}).get('backup_monthly', 0)
                    transfer_cost = prod_result.get('cost_breakdown', {}).get('transfer_monthly', 50)
                    
                    if writer_cost > 0:
                        cost_data['Writer Instance'] = writer_cost
                    if readers_cost > 0:
                        cost_data['Reader Instances'] = readers_cost
                    if storage_cost > 0:
                        cost_data['Storage'] = storage_cost
                    if backup_cost > 0:
                        cost_data['Backup'] = backup_cost
                    if transfer_cost > 0:
                        cost_data['Data Transfer'] = transfer_cost
                else:
                    # Standard RDS setup
                    cost_breakdown = prod_result.get('cost_breakdown', {})
                    instance_cost = cost_breakdown.get('instance_monthly', prod_result.get('instance_cost', 0))
                    storage_cost = cost_breakdown.get('storage_monthly', prod_result.get('storage_cost', 0))
                    backup_cost = cost_breakdown.get('backup_monthly', storage_cost * 0.1)
                    transfer_cost = cost_breakdown.get('transfer_monthly', 50)
                    
                    if instance_cost > 0:
                        cost_data['Compute Instance'] = instance_cost
                    if storage_cost > 0:
                        cost_data['Storage'] = storage_cost
                    if backup_cost > 0:
                        cost_data['Backup'] = backup_cost
                    if transfer_cost > 0:
                        cost_data['Data Transfer'] = transfer_cost
                
                # Filter out zero values and ensure we have data
                cost_data = {k: v for k, v in cost_data.items() if v > 0}
                
                if not cost_data:
                    # Fallback data if no breakdown available
                    total_cost = prod_result.get('total_cost', 0)
                    cost_data = {
                        'Database Instance': total_cost * 0.7,
                        'Storage & Backup': total_cost * 0.25,
                        'Data Transfer': total_cost * 0.05
                    }
                
                # Create improved pie chart
                colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(cost_data.keys()),
                    values=list(cost_data.values()),
                    hole=.4,
                    textinfo='label+percent',
                    textposition='outside',
                    texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}',
                    textfont={"size": 10},
                    marker=dict(colors=colors_list[:len(cost_data)], line=dict(color='#FFFFFF', width=2))
                )])
                
                fig.update_layout(
                    title={
                        'text': f'Monthly Cost Breakdown - ${sum(cost_data.values()):,.0f}',
                        'x': 0.5,
                        'font': {'size': 14, 'color': 'darkblue'},
                        'y': 0.95
                    },
                    font=dict(size=10),
                    showlegend=True,
                    legend=dict(
                        orientation="v", 
                        yanchor="middle", 
                        y=0.5, 
                        xanchor="left", 
                        x=1.05,
                        font=dict(size=9)
                    ),
                    margin=dict(t=50, b=50, l=50, r=100),
                    width=600,
                    height=400
                )
                
                return self.create_plotly_chart_image(fig, width=600, height=400)
            
            else:  # Bulk analysis
                server_costs = []
                server_names = []
                
                for server_name, server_results in analysis_results.items():
                    if 'error' not in server_results:
                        result = server_results.get('PROD', list(server_results.values())[0])
                        if 'error' not in result:
                            cost = result.get('total_cost', 0)
                            if cost > 0:
                                server_costs.append(cost)
                                # Truncate long server names for better display
                                display_name = server_name[:15] + '...' if len(server_name) > 15 else server_name
                                server_names.append(display_name)
                
                if not server_costs:
                    return None
                
                # Create improved bar chart for bulk analysis
                fig = go.Figure(data=[go.Bar(
                    x=server_names,
                    y=server_costs,
                    marker_color='lightblue',
                    text=[f'${cost:,.0f}' for cost in server_costs],
                    textposition='outside',
                    textfont=dict(size=9)
                )])
                
                fig.update_layout(
                    title={
                        'text': f'Monthly Cost by Server - Total: ${sum(server_costs):,.0f}',
                        'x': 0.5,
                        'font': {'size': 14, 'color': 'darkblue'}
                    },
                    xaxis_title='Server Name',
                    yaxis_title='Monthly Cost ($)',
                    font=dict(size=10),
                    xaxis={'tickangle': 45, 'tickfont': {'size': 8}},
                    margin=dict(t=50, b=80, l=70, r=50),
                    width=700,
                    height=400
                )
                
                return self.create_plotly_chart_image(fig, width=700, height=400)
                
        except Exception as e:
            print(f"Error creating cost chart: {e}")
            return None
    
    def create_plotly_chart_image(self, fig, width=600, height=400):
        """Convert Plotly figure to image for PDF inclusion with improved quality"""
        try:
            # Convert plotly figure to image bytes with high quality
            img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
            
            # Create a temporary file to store the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_bytes)
                tmp_file.flush()
                
                # Create ReportLab Image object with appropriate sizing
                img = Image(tmp_file.name, width=self.chart_width, height=self.chart_height)
                return img
        except Exception as e:
            print(f"Error creating chart image: {e}")
            return None
    
    def create_improved_technical_table(self, analysis_results, server_specs, env_name):
        """Create an improved technical specifications table"""
        try:
            result = analysis_results.get(env_name, {})
            if 'error' in result:
                return None
            
            # Table data with better formatting
            table_data = [
                ['Component', 'Current (On-Premises)', 'Recommended (AWS)', 'Improvement']
            ]
            
            current_cpu = server_specs.get('cores', 0) if server_specs else 0
            current_ram = server_specs.get('ram', 0) if server_specs else 0
            current_storage = server_specs.get('storage', 0) if server_specs else 0
            
            if 'writer' in result:
                # Aurora/RDS Cluster configuration
                writer = result['writer']
                recommended_cpu = writer.get('actual_vCPUs', 0)
                recommended_ram = writer.get('actual_RAM_GB', 0)
                recommended_storage = result.get('storage_GB', 0)
                
                table_data.extend([
                    ['Writer Instance', 
                     f"{current_cpu} cores", 
                     f"{writer.get('instance_type', 'N/A')}", 
                     f"{(recommended_cpu/max(current_cpu,1)):.1f}x CPU"],
                    ['Writer Memory', 
                     f"{current_ram} GB", 
                     f"{recommended_ram} GB", 
                     f"{(recommended_ram/max(current_ram,1)):.1f}x RAM"],
                    ['Storage Capacity', 
                     f"{current_storage} GB", 
                     f"{recommended_storage} GB", 
                     f"{(recommended_storage/max(current_storage,1)):.1f}x"],
                ])
                
                if result.get('readers'):
                    table_data.append([
                        'Read Replicas', 
                        'None', 
                        f"{len(result['readers'])} instances", 
                        'New capability'
                    ])
            
            else:
                # Standard RDS configuration
                recommended_cpu = result.get('actual_vCPUs', 0)
                recommended_ram = result.get('actual_RAM_GB', 0)
                recommended_storage = result.get('storage_GB', 0)
                
                table_data.extend([
                    ['Instance Type', 
                     'Physical Server', 
                     result.get('instance_type', 'N/A'), 
                     'Cloud Native'],
                    ['CPU Cores', 
                     f"{current_cpu} cores", 
                     f"{recommended_cpu} vCPUs", 
                     f"{(recommended_cpu/max(current_cpu,1)):.1f}x"],
                    ['Memory', 
                     f"{current_ram} GB", 
                     f"{recommended_ram} GB", 
                     f"{(recommended_ram/max(current_ram,1)):.1f}x"],
                    ['Storage', 
                     f"{current_storage} GB", 
                     f"{recommended_storage} GB", 
                     f"{(recommended_storage/max(current_storage,1)):.1f}x"]
                ])
            
            # Create table with improved styling
            col_widths = [1.8*inch, 1.6*inch, 1.8*inch, 1.2*inch]
            tech_table = Table(table_data, colWidths=col_widths, repeatRows=1)
            
            # Enhanced table styling
            tech_table.setStyle(TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, 0), 12),
                
                # Data rows styling
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white]),
                
                # Left align the first column
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                
                # Padding for better readability
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ]))
            
            return tech_table
            
        except Exception as e:
            print(f"Error creating technical table for {env_name}: {e}")
            return None
    
    def format_ai_insights_text(self, ai_text, max_line_length=80):
        """Format AI insights text for better PDF display"""
        if not ai_text:
            return "No AI insights available."
        
        # Clean up the text
        cleaned_text = ai_text.replace('...', '')
        
        # Split into sentences and paragraphs
        sentences = cleaned_text.split('. ')
        formatted_paragraphs = []
        current_paragraph = ""
        
        for sentence in sentences:
            if len(current_paragraph) + len(sentence) < max_line_length:
                current_paragraph += sentence + ". "
            else:
                if current_paragraph:
                    formatted_paragraphs.append(current_paragraph.strip())
                current_paragraph = sentence + ". "
        
        if current_paragraph:
            formatted_paragraphs.append(current_paragraph.strip())
        
        return formatted_paragraphs[:5]  # Limit to 5 paragraphs for PDF
    
    def generate_improved_pdf_report(self, analysis_results, analysis_mode, server_specs=None, ai_insights=None, transfer_results=None):
        """Generate improved PDF report with better formatting and layout"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=0.75*inch, 
            leftMargin=0.75*inch,
            topMargin=1*inch, 
            bottomMargin=1*inch
        )
        
        story = []
        
        try:
            # Title Page
            story.extend(self._create_title_page(analysis_mode, analysis_results))
            story.append(PageBreak())
            
            # Executive Summary with Chart
            story.extend(self._create_executive_summary(analysis_results, analysis_mode, ai_insights))
            story.append(PageBreak())
            
            # Technical Analysis
            story.extend(self._create_technical_analysis(analysis_results, analysis_mode, server_specs))
            story.append(PageBreak())
            
            # Financial Analysis
            story.extend(self._create_financial_analysis(analysis_results, analysis_mode))
            story.append(PageBreak())
            
            # Migration Strategy
            story.extend(self._create_migration_strategy())
            story.append(PageBreak())
            
            # AI Insights (if available)
            if ai_insights:
                story.extend(self._create_ai_insights_section(ai_insights))
                story.append(PageBreak())
            
            # Implementation Roadmap
            story.extend(self._create_implementation_roadmap())
            
            # Build the PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error building improved PDF: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_title_page(self, analysis_mode, analysis_results):
        """Create an improved title page"""
        story = []
        
        # Main title
        title_style = self.safe_get_style('ReportTitle', 'Title')
        story.append(Paragraph("AWS RDS Migration & Sizing", title_style))
        story.append(Paragraph("Comprehensive Analysis Report", title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Analysis type
        analysis_type = "Single Server Analysis" if analysis_mode == 'single' else "Bulk Server Analysis"
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph(analysis_type, section_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                monthly_cost = prod_result.get('total_cost', 0)
                
                summary_items = [
                    f"Monthly Cloud Cost: ${monthly_cost:,.2f}",
                    f"Annual Investment: ${monthly_cost * 12:,.2f}",
                    "Migration Type: Heterogeneous Database Migration",
                    "Estimated Timeline: 12-16 weeks",
                    "Risk Level: Medium (Manageable with proper planning)"
                ]
        else:
            total_servers = len(analysis_results)
            successful_servers = sum(1 for result in analysis_results.values() if 'error' not in result)
            total_monthly_cost = sum(
                result.get('PROD', {}).get('total_cost', 0) 
                for result in analysis_results.values() 
                if 'error' not in result and 'PROD' in result
            )
            
            summary_items = [
                f"Total Servers Analyzed: {total_servers}",
                f"Successful Analyses: {successful_servers}",
                f"Total Monthly Cost: ${total_monthly_cost:,.2f}",
                f"Total Annual Investment: ${total_monthly_cost * 12:,.2f}",
                f"Average Cost per Server: ${total_monthly_cost/max(successful_servers,1):,.2f}/month"
            ]
        
        # Summary box
        summary_style = self.safe_get_style('KeyMetric', 'Normal')
        story.append(Paragraph("<b>Executive Summary</b>", summary_style))
        
        bullet_style = self.safe_get_style('BulletPoint', 'Normal')
        for item in summary_items:
            story.append(Paragraph(f"• {item}", bullet_style))
        
        story.append(Spacer(1, 0.5*inch))
        
        # Report metadata
        body_style = self.safe_get_style('BodyText', 'Normal')
        generation_time = datetime.now().strftime("%B %d, %Y at %H:%M")
        story.append(Paragraph(f"<b>Generated:</b> {generation_time}", body_style))
        story.append(Paragraph("<b>Report Type:</b> Comprehensive Technical & Financial Analysis", body_style))
        story.append(Paragraph("<b>Prepared for:</b> Enterprise Cloud Migration Team", body_style))
        
        return story
    
    def _create_executive_summary(self, analysis_results, analysis_mode, ai_insights):
        """Create improved executive summary with chart"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Executive Summary & Key Findings", section_style))
        story.append(Spacer(1, 12))
        
        # Cost breakdown chart
        cost_chart = self.create_improved_cost_chart(analysis_results, analysis_mode)
        if cost_chart:
            story.append(cost_chart)
            story.append(Spacer(1, 20))
        
        # Key findings
        subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
        story.append(Paragraph("Key Findings & Recommendations:", subsection_style))
        
        body_style = self.safe_get_style('BodyText', 'Normal')
        bullet_style = self.safe_get_style('BulletPoint', 'Normal')
        
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            if valid_results:
                prod_result = valid_results.get('PROD', list(valid_results.values())[0])
                monthly_cost = prod_result.get('total_cost', 0)
                
                findings = [
                    f"<b>Cost Efficiency:</b> Projected monthly operational cost of ${monthly_cost:,.2f} provides significant operational benefits over traditional on-premises infrastructure",
                    "<b>Performance Scaling:</b> AWS RDS configuration will provide improved performance and automatic scaling capabilities",
                    "<b>Risk Mitigation:</b> Multi-AZ deployment ensures 99.95% uptime SLA with automatic failover",
                    "<b>Operational Benefits:</b> Reduced maintenance overhead with managed database services"
                ]
        else:
            successful_servers = sum(1 for result in analysis_results.values() if 'error' not in result)
            total_monthly_cost = sum(
                result.get('PROD', {}).get('total_cost', 0) 
                for result in analysis_results.values() 
                if 'error' not in result and 'PROD' in result
            )
            
            findings = [
                f"<b>Scale Efficiency:</b> {successful_servers} servers successfully analyzed with total monthly cost of ${total_monthly_cost:,.2f}",
                "<b>Migration Approach:</b> Phased migration recommended with 3-5 servers per wave",
                "<b>Cost Optimization:</b> Bulk Reserved Instance purchases can reduce costs by 30-40%",
                "<b>Timeline:</b> Estimated 6-9 months for complete bulk migration with parallel streams"
            ]
        
        for finding in findings:
            story.append(Paragraph(f"• {finding}", bullet_style))
            story.append(Spacer(1, 6))
        
        return story
    
    def _create_technical_analysis(self, analysis_results, analysis_mode, server_specs):
        """Create improved technical analysis section"""
        story = []
        
        section_style = self.safe_get_style('SectionHeader', 'Heading1')
        story.append(Paragraph("Technical Analysis & Performance Assessment", section_style))
        story.append(Spacer(1, 15))
        
        if analysis_mode == 'single':
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            
            for env, result in valid_results.items():
                subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
                story.append(Paragraph(f"{env} Environment Configuration", subsection_style))
                story.append(Spacer(1, 10))
                
                tech_table = self.create_improved_technical_table(analysis_results, server_specs, env)
                if tech_table:
                    story.append(tech_table)
                    story.append(Spacer(1, 20))
        
        else:
            # Bulk analysis summary table
            story.extend(self._create_bulk_technical_summary(analysis_results, server_specs))
        
        return story
    
    def _create_bulk_technical_summary(self, analysis_results, server_specs):
        """Create bulk technical analysis summary"""
        story = []
        
        # Aggregate statistics
        total_current_cpu = sum(server.get('cpu_cores', 0) for server in server_specs) if server_specs else 0
        total_current_ram = sum(server.get('ram_gb', 0) for server in server_specs) if server_specs else 0
        total_current_storage = sum(server.get('storage_gb', 0) for server in server_specs) if server_specs else 0        
        total_recommended_cpu = 0
        total_recommended_ram = 0
        total_recommended_storage = 0
        successful_count = 0        
        for server_results in analysis_results.values():
                if 'error' not in server_results:
                    result = server_results.get('PROD', list(server_results.values())[0])
                    if 'error' not in result:
                        successful_count += 1
                        if 'writer' in result:
                            writer = result['writer']
                            total_recommended_cpu += writer.get('actual_vCPUs', 0)
                            total_recommended_ram += writer.get('actual_RAM_GB', 0)
                        else:
                            total_recommended_cpu += result.get('actual_vCPUs', 0)
                            total_recommended_ram += result.get('actual_RAM_GB', 0)
                            total_recommended_storage += result.get('storage_GB', 0)
       
       # Create aggregate comparison table
        subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
        story.append(Paragraph("Aggregate Resource Comparison", subsection_style))
        story.append(Spacer(1, 10))
            
        aggregate_data = [
                ['Resource Type', 'Current Total', 'Recommended Total', 'Efficiency Gain'],
                ['Total vCPUs', f"{total_current_cpu}", f"{total_recommended_cpu}", f"{((total_recommended_cpu/max(total_current_cpu,1))-1)*100:+.1f}%"],
                ['Total RAM (GB)', f"{total_current_ram:,}", f"{total_recommended_ram:,}", f"{((total_recommended_ram/max(total_current_ram,1))-1)*100:+.1f}%"],
                ['Total Storage (GB)', f"{total_current_storage:,}", f"{total_recommended_storage:,}", f"{((total_recommended_storage/max(total_current_storage,1))-1)*100:+.1f}%"],
                ['Server Count', f"{len(server_specs) if server_specs else 0}", f"{successful_count}", f"{(successful_count/max(len(server_specs) if server_specs else 1,1))*100:.0f}% success"]
                        ]
            
        aggregate_table = Table(aggregate_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        aggregate_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
        story.append(aggregate_table)        
        return story
   
def _create_financial_analysis(self, analysis_results, analysis_mode):
       """Create improved financial analysis section"""
       story = []
       
       section_style = self.safe_get_style('SectionHeader', 'Heading1')
       story.append(Paragraph("Financial Analysis & ROI Projection", section_style))
       story.append(Spacer(1, 15))
       
       # Calculate costs
       if analysis_mode == 'single':
           valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
           if valid_results:
               prod_result = valid_results.get('PROD', list(valid_results.values())[0])
               monthly_cost = prod_result.get('total_cost', 0)
               
               financial_data = [
                   ['Cost Component', 'Monthly', 'Annual', '3-Year Total'],
                   ['AWS Infrastructure', f'${monthly_cost:,.2f}', f'${monthly_cost*12:,.2f}', f'${monthly_cost*36:,.2f}'],
                   ['Migration Costs (One-time)', '-', f'${monthly_cost*2:,.2f}', f'${monthly_cost*2:,.2f}'],
                   ['Training & Support', '-', f'${monthly_cost*0.5:,.2f}', f'${monthly_cost*1.5:,.2f}'],
                   ['Total Investment', f'${monthly_cost:,.2f}', f'${monthly_cost*14.5:,.2f}', f'${monthly_cost*39.5:,.2f}']
               ]
       else:
           total_monthly_cost = sum(
               result.get('PROD', {}).get('total_cost', 0) 
               for result in analysis_results.values() 
               if 'error' not in result and 'PROD' in result
           )
           
           financial_data = [
               ['Cost Component', 'Monthly', 'Annual', '3-Year Total'],
               ['AWS Infrastructure', f'${total_monthly_cost:,.2f}', f'${total_monthly_cost*12:,.2f}', f'${total_monthly_cost*36:,.2f}'],
               ['Migration Costs (One-time)', '-', f'${total_monthly_cost*3:,.2f}', f'${total_monthly_cost*3:,.2f}'],
               ['Training & Support', '-', f'${total_monthly_cost*1:,.2f}', f'${total_monthly_cost*3:,.2f}'],
               ['Total Investment', f'${total_monthly_cost:,.2f}', f'${total_monthly_cost*16:,.2f}', f'${total_monthly_cost*42:,.2f}']
           ]
       
       # Create financial table
       financial_table = Table(financial_data, colWidths=[2.2*inch, 1.4*inch, 1.4*inch, 1.4*inch])
       financial_table.setStyle(TableStyle([
           ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
           ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
           ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
           ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
           ('FONTSIZE', (0, 0), (-1, 0), 11),
           ('FONTSIZE', (0, 1), (-1, -1), 10),
           ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
           ('BACKGROUND', (0, 1), (-1, -2), colors.mistyrose),
           ('BACKGROUND', (0, -1), (-1, -1), colors.yellow),
           ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
           ('GRID', (0, 0), (-1, -1), 1, colors.black),
           ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
           ('TOPPADDING', (0, 0), (-1, -1), 8),
           ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
       ]))
       story.append(financial_table)
       
       return story
   
def _create_migration_strategy(self):
       """Create migration strategy section"""
       story = []
       
       section_style = self.safe_get_style('SectionHeader', 'Heading1')
       story.append(Paragraph("Migration Strategy & Implementation Timeline", section_style))
       story.append(Spacer(1, 15))
       
       subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
       story.append(Paragraph("Migration Strategy Overview:", subsection_style))
       
       bullet_style = self.safe_get_style('BulletPoint', 'Normal')
       migration_phases = [
           "<b>Assessment Phase:</b> Complete discovery and compatibility analysis",
           "<b>Planning Phase:</b> Detailed migration planning and resource allocation",
           "<b>Execution Phase:</b> Phased migration with minimal downtime",
           "<b>Validation Phase:</b> Comprehensive testing and performance validation",
           "<b>Optimization Phase:</b> Post-migration tuning and optimization"
       ]
       
       for phase in migration_phases:
           story.append(Paragraph(f"• {phase}", bullet_style))
           story.append(Spacer(1, 6))
       
       return story
   
def _create_ai_insights_section(self, ai_insights):
       """Create improved AI insights section"""
       story = []
       
       section_style = self.safe_get_style('SectionHeader', 'Heading1')
       story.append(Paragraph("AI-Powered Insights & Recommendations", section_style))
       story.append(Spacer(1, 15))
       
       # AI analysis
       ai_text = ai_insights.get("ai_analysis", "")
       if ai_text:
           formatted_paragraphs = self.format_ai_insights_text(ai_text)
           
           body_style = self.safe_get_style('BodyText', 'Normal')
           for paragraph in formatted_paragraphs:
               story.append(Paragraph(paragraph, body_style))
               story.append(Spacer(1, 10))
       else:
           # Default AI insights
           bullet_style = self.safe_get_style('BulletPoint', 'Normal')
           default_insights = [
               "Migration analysis indicates favorable conditions for AWS RDS migration",
               "Recommended architecture provides optimal balance of performance and cost",
               "Risk assessment suggests manageable migration with proper planning",
               "Cost optimization opportunities identified for long-term efficiency"
           ]
           
           for insight in default_insights:
               story.append(Paragraph(f"• {insight}", bullet_style))
               story.append(Spacer(1, 6))
       
       return story
   
def _create_implementation_roadmap(self):
       """Create implementation roadmap section"""
       story = []
       
       section_style = self.safe_get_style('SectionHeader', 'Heading1')
       story.append(Paragraph("Implementation Roadmap & Next Steps", section_style))
       story.append(Spacer(1, 15))
       
       subsection_style = self.safe_get_style('SubsectionHeader', 'Heading2')
       story.append(Paragraph("Implementation Roadmap:", subsection_style))
       
       bullet_style = self.safe_get_style('BulletPoint', 'Normal')
       roadmap_phases = [
           "<b>Phase 1 - Planning:</b> Finalize migration strategy and resource allocation",
           "<b>Phase 2 - Preparation:</b> Set up AWS environment and migration tools",
           "<b>Phase 3 - Migration:</b> Execute data and application migration",
           "<b>Phase 4 - Testing:</b> Comprehensive validation and performance testing",
           "<b>Phase 5 - Go-Live:</b> Production cutover and monitoring",
           "<b>Phase 6 - Optimization:</b> Post-migration tuning and optimization"
       ]
       
       for phase in roadmap_phases:
           story.append(Paragraph(f"• {phase}", bullet_style))
           story.append(Spacer(1, 6))
       
       story.append(Spacer(1, 20))
       
       # Success criteria
       story.append(Paragraph("Success Criteria:", subsection_style))
       
       success_criteria = [
           "Zero data loss during migration",
           "Minimal downtime during cutover",
           "Performance meets or exceeds baseline",
           "Cost targets achieved within budget",
           "Team readiness and knowledge transfer complete"
       ]
       
       for criteria in success_criteria:
           story.append(Paragraph(f"• {criteria}", bullet_style))
           story.append(Spacer(1, 6))
       
       return story


# Helper function to use the improved generator
def generate_improved_pdf_report(analysis_results, analysis_mode, server_specs=None, ai_insights=None, transfer_results=None):
   """Helper function with improved formatting and error handling"""
   try:
       print("Creating ImprovedReportGenerator...")
       improved_generator = ImprovedReportGenerator()
       
       print("Validating inputs...")
       if not analysis_results:
           print("Error: No analysis results provided")
           return None
       
       if analysis_mode not in ['single', 'bulk']:
           print(f"Error: Invalid analysis mode '{analysis_mode}'")
           return None
       
       print("Generating improved PDF...")
       pdf_bytes = generate_improved_pdf_report(
        analysis_results=current_analysis_results,
        analysis_mode=analysis_mode_for_pdf,
        server_specs=current_server_specs_for_pdf,
        ai_insights=ai_insights_for_pdf,
        transfer_results=transfer_results_for_pdf
         )
       
       if pdf_bytes:
           print(f"Improved PDF generated successfully. Size: {len(pdf_bytes)} bytes")
           return pdf_bytes
       else:
           print("Error: PDF generation returned None")
           return None
       
   except Exception as e:
       print(f"Error in improved PDF generation: {str(e)}")
       import traceback
       traceback.print_exc()
       return None

# ALTERNATIVE: Cached version for better performance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_comprehensive_pdf_report_cached(analysis_results_str, analysis_mode, server_specs_str=None, ai_insights_str=None, transfer_results_str=None):
   """
   Cached version of PDF generator (converts strings back to objects)
   Use this if you're generating the same report multiple times
   """
   import json
   
   try:
       # Convert string parameters back to objects
       analysis_results = json.loads(analysis_results_str) if analysis_results_str else None
       server_specs = json.loads(server_specs_str) if server_specs_str else None
       ai_insights = json.loads(ai_insights_str) if ai_insights_str else None
       transfer_results = json.loads(transfer_results_str) if transfer_results_str else None
       
       # Call the main function
       return generate_improved_pdf_report(
           analysis_results=analysis_results,
           analysis_mode=analysis_mode,
           server_specs=server_specs,
           ai_insights=ai_insights,
           transfer_results=transfer_results
       )
   
   except json.JSONDecodeError as e:
       print(f"Error parsing cached parameters: {e}")
       return None


class DatabaseEngine:
    """Database engine definitions and compatibility matrix"""
    
    ENGINES = {
        'oracle-ee': {
            'name': 'Oracle Enterprise Edition',
            'features': ['Advanced Analytics', 'Partitioning', 'Advanced Security', 'RAC'],
            'aws_targets': ['oracle-ee', 'postgres', 'aurora-postgresql'],
            'complexity_multiplier': 1.0
        },
        'oracle-se': {
            'name': 'Oracle Standard Edition',
            'features': ['Basic Analytics', 'Standard Security'],
            'aws_targets': ['oracle-se', 'postgres', 'aurora-postgresql'],
            'complexity_multiplier': 0.8
        },
        'postgres': {
            'name': 'PostgreSQL',
            'features': ['JSON Support', 'Advanced Indexing', 'Extensions'],
            'aws_targets': ['postgres', 'aurora-postgresql'],
            'complexity_multiplier': 0.5
        },
        'mysql': {
            'name': 'MySQL',
            'features': ['InnoDB', 'MyISAM', 'Replication'],
            'aws_targets': ['mysql', 'aurora-mysql'],
            'complexity_multiplier': 0.6
        },
        'sql-server': {
            'name': 'Microsoft SQL Server',
            'features': ['T-SQL', 'SSRS', 'SSIS', 'Analysis Services'],
            'aws_targets': ['sql-server', 'postgres', 'aurora-postgresql'],
            'complexity_multiplier': 1.2
        },
        'mariadb': {
            'name': 'MariaDB',
            'features': ['MySQL Compatibility', 'Columnar Storage'],
            'aws_targets': ['mariadb', 'mysql', 'aurora-mysql'],
            'complexity_multiplier': 0.4
        },
        'aurora-postgresql': {
            'name': 'Amazon Aurora PostgreSQL',
            'features': ['Auto-scaling', 'Global Database', 'Serverless'],
            'aws_targets': ['aurora-postgresql'],
            'complexity_multiplier': 0.3
        },
        'aurora-mysql': {
            'name': 'Amazon Aurora MySQL',
            'features': ['Auto-scaling', 'Global Database', 'Serverless'],
            'aws_targets': ['aurora-mysql'],
            'complexity_multiplier': 0.3
        }
    }
    
    @classmethod
    def get_migration_type(cls, source_engine: str, target_engine: str) -> str:
        """Determine migration type based on source and target engines"""
        
        if source_engine == target_engine:
            return "homogeneous"
        
        # Check if engines are in the same family
        mysql_family = ['mysql', 'mariadb', 'aurora-mysql']
        postgres_family = ['postgres', 'aurora-postgresql']
        oracle_family = ['oracle-ee', 'oracle-se']
        
        source_family = None
        target_family = None
        
        for family in [mysql_family, postgres_family, oracle_family]:
            if source_engine in family:
                source_family = family
            if target_engine in family:
                target_family = family
        
        if source_family and source_family == target_family:
            return "homogeneous"
        else:
            return "heterogeneous"
    
    @classmethod
    def get_complexity_multiplier(cls, source_engine: str, target_engine: str) -> float:
        """Get complexity multiplier for migration"""
        
        if source_engine == target_engine:
            return 1.0
        
        # Base complexity from source engine
        source_complexity = cls.ENGINES.get(source_engine, {}).get('complexity_multiplier', 1.0)
        target_complexity = cls.ENGINES.get(target_engine, {}).get('complexity_multiplier', 1.0)
        
        # Additional complexity for heterogeneous migrations
        migration_type = cls.get_migration_type(source_engine, target_engine)
        
        if migration_type == "heterogeneous":
            if 'oracle' in source_engine and 'postgres' in target_engine:
                return source_complexity * 2.5  # Oracle to PostgreSQL is complex
            elif 'sql-server' in source_engine and 'postgres' in target_engine:
                return source_complexity * 2.0  # SQL Server to PostgreSQL
            else:
                return source_complexity * 1.5  # Other heterogeneous migrations
        else:
            return max(source_complexity, target_complexity)
    
    @classmethod
    def get_supported_features(cls, engine: str) -> list:
        """Get supported features for an engine"""
        return cls.ENGINES.get(engine, {}).get('features', [])
    
    @classmethod
    def get_aws_targets(cls, source_engine: str) -> list:
        """Get available AWS target engines for a source engine"""
        return cls.ENGINES.get(source_engine, {}).get('aws_targets', [])
# Page Configuration
st.set_page_config(
    page_title="Enterprise AWS Database Migration Tool",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Enhanced Enterprise CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .enterprise-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .enterprise-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .config-section {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #4299e1;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0;
    }
    
    .environment-card {
        background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid #38b2ac;
        margin: 1rem 0;
    }
    
    .ai-insight-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    
    .risk-high { border-left-color: #e53e3e !important; }
    .risk-medium { border-left-color: #d69e2e !important; }
    .risk-low { border-left-color: #38a169 !important; }
    
    .strategy-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class GrowthAwareCostAnalyzer:
    """Cost analyzer with 3-year growth projections"""
    
    def __init__(self):
        self.growth_factors = self._initialize_growth_factors()
    
    def _initialize_growth_factors(self):
        """Initialize growth impact factors for different cost components"""
        return {
            'compute': {
                'user_growth_factor': 0.8,      # 80% correlation with user growth
                'transaction_factor': 0.9,      # 90% correlation with transactions
                'data_factor': 0.3               # 30% correlation with data size
            },
            'storage': {
                'data_growth_factor': 1.0,      # 100% correlation with data growth
                'user_growth_factor': 0.2,      # 20% correlation with users
                'transaction_factor': 0.4       # 40% correlation with transactions
            },
            'iops': {
                'transaction_factor': 0.95,     # 95% correlation with transactions
                'user_growth_factor': 0.7,      # 70% correlation with users
                'data_factor': 0.5               # 50% correlation with data size
            },
            'network': {
                'user_growth_factor': 0.8,      # 80% correlation with users
                'transaction_factor': 0.6,      # 60% correlation with transactions
                'data_factor': 0.4               # 40% correlation with data size
            }
        }
    
    def calculate_3_year_growth_projection(self, base_costs: Dict, migration_params: Dict) -> Dict:
        """Calculate 3-year cost projection with growth"""
        
        # Extract growth parameters
        data_growth = migration_params.get('annual_data_growth', 15) / 100
        user_growth = migration_params.get('annual_user_growth', 25) / 100
        transaction_growth = migration_params.get('annual_transaction_growth', 20) / 100
        seasonality = migration_params.get('seasonality_factor', 1.2)
        scaling_strategy = migration_params.get('scaling_strategy', 'Auto-scaling')
        
        projections = {}
        
        for year in range(4):  # Year 0 (current) through Year 3
            year_multiplier = {
                'data': (1 + data_growth) ** year,
                'users': (1 + user_growth) ** year,
                'transactions': (1 + transaction_growth) ** year
            }
            
            year_costs = self._calculate_year_costs(
                base_costs, year_multiplier, seasonality, scaling_strategy, year
            )
            
            projections[f'year_{year}'] = year_costs
        
        # Calculate scaling recommendations
        scaling_recommendations = self._generate_scaling_recommendations(
            projections, migration_params
        )
        
        return {
            'yearly_projections': projections,
            'scaling_recommendations': scaling_recommendations,
            'growth_summary': self._generate_growth_summary(projections),
            'cost_optimization_opportunities': self._identify_growth_optimizations(projections)
        }
    
    def _calculate_year_costs(self, base_costs: Dict, multipliers: Dict, 
                            seasonality: float, scaling_strategy: str, year: int) -> Dict:
        """Calculate costs for a specific year with growth factors"""
        
        year_costs = {}
        
        for env_name, env_costs in base_costs['environment_costs'].items():
            # Calculate growth-adjusted resource requirements
            compute_multiplier = (
                multipliers['users'] * self.growth_factors['compute']['user_growth_factor'] +
                multipliers['transactions'] * self.growth_factors['compute']['transaction_factor'] +
                multipliers['data'] * self.growth_factors['compute']['data_factor']
            ) / 3
            
            storage_multiplier = (
                multipliers['data'] * self.growth_factors['storage']['data_growth_factor'] +
                multipliers['users'] * self.growth_factors['storage']['user_growth_factor'] +
                multipliers['transactions'] * self.growth_factors['storage']['transaction_factor']
            ) / 3
            
            iops_multiplier = (
                multipliers['transactions'] * self.growth_factors['iops']['transaction_factor'] +
                multipliers['users'] * self.growth_factors['iops']['user_growth_factor'] +
                multipliers['data'] * self.growth_factors['iops']['data_factor']
            ) / 3
            
            # Apply scaling strategy adjustments
            if scaling_strategy == "Over-provision":
                compute_multiplier *= 1.3  # 30% over-provisioning
                iops_multiplier *= 1.2     # 20% over-provisioning
            elif scaling_strategy == "Auto-scaling":
                # More gradual scaling with auto-scaling
                compute_multiplier = min(compute_multiplier, 1 + (year * 0.5))
            
            # Apply seasonality to peak requirements
            peak_compute = compute_multiplier * seasonality
            peak_iops = iops_multiplier * seasonality
            
            # Calculate adjusted costs
            base_instance_cost = env_costs.get('instance_cost', env_costs.get('writer_instance_cost', 0))
            base_storage_cost = env_costs.get('storage_cost', 0)
            base_reader_cost = env_costs.get('reader_costs', 0)
            
            adjusted_costs = {
                'instance_cost': base_instance_cost * compute_multiplier,
                'storage_cost': base_storage_cost * storage_multiplier,
                'iops_cost': (base_storage_cost * 0.3) * iops_multiplier,  # Estimate IOPS as 30% of storage
                'reader_costs': base_reader_cost * compute_multiplier,
                'backup_cost': (base_storage_cost * storage_multiplier) * 0.2,
                'monitoring_cost': 50 * compute_multiplier,  # Base monitoring cost
                'total_monthly': 0,
                'peak_monthly': 0,  # Peak cost with seasonality
                'resource_scaling': {
                    'compute_scaling': compute_multiplier,
                    'storage_scaling': storage_multiplier,
                    'iops_scaling': iops_multiplier,
                    'recommended_reader_scaling': max(1, int(compute_multiplier))
                }
            }
            
            adjusted_costs['total_monthly'] = sum([
                adjusted_costs['instance_cost'],
                adjusted_costs['storage_cost'],
                adjusted_costs['iops_cost'],
                adjusted_costs['reader_costs'],
                adjusted_costs['backup_cost'],
                adjusted_costs['monitoring_cost']
            ])
            
            adjusted_costs['peak_monthly'] = adjusted_costs['total_monthly'] * seasonality
            
            year_costs[env_name] = adjusted_costs
        
        # Calculate totals
        total_monthly = sum([env['total_monthly'] for env in year_costs.values()])
        total_peak = sum([env['peak_monthly'] for env in year_costs.values()])
        
        return {
            'environment_costs': year_costs,
            'total_monthly': total_monthly,
            'total_annual': total_monthly * 12,
            'peak_monthly': total_peak,
            'peak_annual': total_peak * 12,
            'year': year,
            'growth_multipliers': multipliers
        }
    
    def _generate_scaling_recommendations(self, projections: Dict, migration_params: Dict) -> List[Dict]:
        """Generate scaling recommendations based on growth projections"""
        
        recommendations = []
        
        # Analyze year-over-year growth
        for year in range(1, 4):
            current_year = projections[f'year_{year}']
            previous_year = projections[f'year_{year-1}']
            
            cost_increase = ((current_year['total_annual'] / previous_year['total_annual']) - 1) * 100
            
            if cost_increase > 50:
                recommendations.append({
                    'year': year,
                    'type': 'Critical Scaling',
                    'description': f"Year {year} shows {cost_increase:.1f}% cost increase",
                    'action': 'Consider Reserved Instances and architecture optimization',
                    'priority': 'High',
                    'estimated_savings': current_year['total_annual'] * 0.3
                })
            elif cost_increase > 25:
                recommendations.append({
                    'year': year,
                    'type': 'Moderate Scaling',
                    'description': f"Year {year} shows {cost_increase:.1f}% cost increase",
                    'action': 'Plan for capacity increases and evaluate read replicas',
                    'priority': 'Medium',
                    'estimated_savings': current_year['total_annual'] * 0.15
                })
        
        # Storage growth recommendations
        year_3_storage_growth = sum([
            env['resource_scaling']['storage_scaling'] 
            for env in projections['year_3']['environment_costs'].values()
        ]) / len(projections['year_3']['environment_costs'])
        
        if year_3_storage_growth > 3:
            recommendations.append({
                'year': 3,
                'type': 'Storage Optimization',
                'description': f"Storage requirements projected to grow {year_3_storage_growth:.1f}x",
                'action': 'Implement data lifecycle policies and archiving strategy',
                'priority': 'Medium',
                'estimated_savings': projections['year_3']['total_annual'] * 0.2
            })
        
        return recommendations
    
    def _generate_growth_summary(self, projections: Dict) -> Dict:
        """Generate growth summary statistics"""
        
        year_0 = projections['year_0']['total_annual']
        year_3 = projections['year_3']['total_annual']
        
        total_growth = ((year_3 / year_0) - 1) * 100
        cagr = ((year_3 / year_0) ** (1/3) - 1) * 100
        
        return {
            'total_3_year_growth_percent': total_growth,
            'compound_annual_growth_rate': cagr,
            'year_0_cost': year_0,
            'year_3_cost': year_3,
            'total_3_year_investment': sum([
                projections[f'year_{year}']['total_annual'] for year in range(4)
            ]),
            'average_annual_cost': sum([
                projections[f'year_{year}']['total_annual'] for year in range(4)
            ]) / 4
        }
    
    def _identify_growth_optimizations(self, projections: Dict) -> List[Dict]:
        """Identify cost optimization opportunities based on growth projections"""
        
        optimizations = []
        
        year_0 = projections['year_0']
        year_3 = projections['year_3']
        
        # Calculate growth rates for different cost components
        total_growth = (year_3['total_annual'] / year_0['total_annual'] - 1) * 100
        
        # Storage optimization opportunities
        storage_growth = 0
        compute_growth = 0
        env_count = len(year_0['environment_costs'])
        
        for env_name in year_0['environment_costs'].keys():
            year_0_env = year_0['environment_costs'][env_name]
            year_3_env = year_3['environment_costs'][env_name]
            
            storage_growth += year_3_env['resource_scaling']['storage_scaling']
            compute_growth += year_3_env['resource_scaling']['compute_scaling']
        
        avg_storage_growth = storage_growth / env_count
        avg_compute_growth = compute_growth / env_count
        
        # Generate optimization recommendations
        if avg_storage_growth > 2.5:
            optimizations.append({
                'opportunity': 'Data Lifecycle Management',
                'description': f"Storage projected to grow {avg_storage_growth:.1f}x over 3 years. Implement automated data archiving and tiering.",
                'estimated_savings': year_3['total_annual'] * 0.15,
                'implementation_effort': 'Medium',
                'timeline': '3-6 months'
            })
        
        if avg_compute_growth > 2.0:
            optimizations.append({
                'opportunity': 'Reserved Instance Strategy',
                'description': f"Compute resources growing {avg_compute_growth:.1f}x. Reserved Instances can provide 30-60% savings.",
                'estimated_savings': year_3['total_annual'] * 0.35,
                'implementation_effort': 'Low',
                'timeline': '1-2 months'
            })
        
        if total_growth > 100:  # More than doubling
            optimizations.append({
                'opportunity': 'Architecture Optimization',
                'description': f"Total costs growing {total_growth:.1f}% over 3 years. Consider microservices architecture and serverless components.",
                'estimated_savings': year_3['total_annual'] * 0.25,
                'implementation_effort': 'High',
                'timeline': '6-12 months'
            })
        
        # Seasonal optimization
        peak_ratio = year_3['peak_annual'] / year_3['total_annual']
        if peak_ratio > 1.3:  # 30% peak difference
            optimizations.append({
                'opportunity': 'Auto-scaling Implementation',
                'description': f"Peak costs are {peak_ratio:.1f}x average. Auto-scaling can reduce costs during low-demand periods.",
                'estimated_savings': (year_3['peak_annual'] - year_3['total_annual']) * 0.6,
                'implementation_effort': 'Medium',
                'timeline': '2-4 months'
            })
        
        # Reader replica optimization
        total_reader_scaling = sum([
            env['resource_scaling']['recommended_reader_scaling'] 
            for env in year_3['environment_costs'].values()
        ])
        
        if total_reader_scaling > env_count * 2:  # More than 2 readers per environment on average
            optimizations.append({
                'opportunity': 'Read Replica Optimization',
                'description': f"Projected to need {total_reader_scaling} total read replicas. Consider Aurora Serverless v2 for variable read workloads.",
                'estimated_savings': year_3['total_annual'] * 0.12,
                'implementation_effort': 'Low',
                'timeline': '1-3 months'
            })
        
        # If no specific optimizations found, provide general recommendations
        if not optimizations:
            optimizations.append({
                'opportunity': 'Continuous Optimization',
                'description': 'Implement regular cost reviews and right-sizing exercises every 6 months.',
                'estimated_savings': year_3['total_annual'] * 0.10,
                'implementation_effort': 'Low',
                'timeline': 'Ongoing'
            })
        
        return optimizations
    
    def _generate_growth_summary(self, projections: Dict) -> Dict:
        """Generate growth summary statistics"""
        
        year_0 = projections['year_0']['total_annual']
        year_3 = projections['year_3']['total_annual']
        
        total_growth = ((year_3 / year_0) - 1) * 100
        cagr = ((year_3 / year_0) ** (1/3) - 1) * 100
        
        return {
            'total_3_year_growth_percent': total_growth,
            'compound_annual_growth_rate': cagr,
            'year_0_cost': year_0,
            'year_3_cost': year_3,
            'total_3_year_investment': sum([
                projections[f'year_{year}']['total_annual'] for year in range(4)
            ]),
            'average_annual_cost': sum([
                projections[f'year_{year}']['total_annual'] for year in range(4)
            ]) / 4
        }


# ===========================
# CORE CLASSES AND FUNCTIONS
# ===========================

# REAL AWS Pricing API Implementation
class RealAWSPricingAPI:
    """Real AWS Pricing API that fetches live pricing data"""
    
    def __init__(self):
        self.base_url = "https://pricing.us-east-1.amazonaws.com"
        self.cache = {}
        # Initialize boto3 pricing client
        try:
            self.pricing_client = boto3.client('pricing', region_name='us-east-1')
        except Exception as e:
            print(f"Warning: Could not initialize AWS pricing client: {e}")
            self.pricing_client = None
    
    # ===========================
# COST RECONCILIATION MODULE  
# ===========================

def reconcile_cost_data_sources():
    """Ensure both cost sources use identical parameters"""
    
    # 1. Standardize time periods
    cost_analysis_period = {
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',
        'timezone': 'UTC'
    }
    
    # 2. Verify AWS region consistency
    aws_region = 'us-east-1'  # Ensure both use same region
    
    # 3. Check currency and billing cycle
    currency = 'USD'
    billing_cycle = 'monthly'  # vs daily/annual
    
    return cost_analysis_period, aws_region, currency, billing_cycle

def map_cost_categories():
    """Map your AI cost categories to AWS Cost Analysis categories"""
    
    cost_mapping = {
        # Your AI Categories -> AWS Cost Analysis Categories
        'instance_cost': 'Amazon RDS Instance',
        'storage_cost': 'Amazon RDS Storage',
        'backup_cost': 'Amazon RDS Backup',
        'data_transfer': 'Data Transfer',
        'monitoring_cost': 'CloudWatch',
        'reader_costs': 'Amazon RDS Read Replica'
    }
    
    return cost_mapping

def standardize_cost_calculations():
    """Align calculation methodologies"""
    
    # 1. Use same pricing model
    pricing_model = {
        'type': 'on_demand',  # vs reserved, spot
        'include_taxes': True,
        'include_credits': False,
        'include_discounts': True
    }
    
    return pricing_model

def calculate_multi_az_cost(base_cost, is_multi_az):
    """Standardize Multi-AZ calculations"""
    if is_multi_az:
        return base_cost * 2.0  # AWS standard multiplier
    return base_cost

def calculate_iops_cost(storage_type, iops, storage_gb):
    """Align IOPS calculations"""
    if storage_type == 'gp3':
        base_iops = 3000
        additional_iops = max(0, iops - base_iops)
        return additional_iops * 0.005
    elif storage_type == 'io2':
        return iops * 0.065
    return 0

class ReconciliationAWSPricingAPI(RealAWSPricingAPI):
    """Enhanced pricing API with cost analysis alignment"""
    
    def __init__(self):
        super().__init__()
        self.include_taxes = True
        self.has_enterprise_discount = False
        self.include_support_costs = True
    
    def get_reconciled_rds_pricing(self, region, engine, instance_class, multi_az=False):
        """Get pricing that matches AWS Cost Analysis exactly"""
        
        try:
            # Get real AWS pricing
            real_pricing = self._fetch_real_aws_pricing(region, engine, instance_class, multi_az)
            
            if real_pricing:
                # Apply cost analysis adjustments
                reconciled_pricing = self._apply_cost_analysis_adjustments(real_pricing)
                return reconciled_pricing
                
        except Exception as e:
            st.warning(f"Using fallback pricing due to: {e}")
            
        return self._get_fallback_pricing(region, engine, instance_class, multi_az)
    
    def _apply_cost_analysis_adjustments(self, pricing):
        """Apply adjustments to match Cost Analysis"""
        
        # 1. Add tax calculations (if applicable)
        if self.include_taxes:
            pricing['hourly'] *= 1.0875  # Example: 8.75% tax
            
        # 2. Apply enterprise discounts
        if self.has_enterprise_discount:
            pricing['hourly'] *= 0.95  # 5% enterprise discount
            
        # 3. Include support costs
        if self.include_support_costs:
            pricing['support_cost'] = pricing['hourly'] * 0.10  # 10% support
            pricing['hourly'] += pricing['support_cost']
        
        pricing['source'] = 'AWS Pricing API (Reconciled)'
        return pricing
    
    
    def get_rds_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool = False) -> Dict:
        """Get real RDS pricing from AWS Pricing API"""
        cache_key = f"{region}_{engine}_{instance_class}_{multi_az}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Try to get real pricing from AWS
            real_pricing = self._fetch_real_aws_pricing(region, engine, instance_class, multi_az)
            if real_pricing:
                self.cache[cache_key] = real_pricing
                return real_pricing
        except Exception as e:
            print(f"Error fetching real AWS pricing: {e}")
        
        # Fallback to static pricing if API fails
        return self._get_fallback_pricing(region, engine, instance_class, multi_az)
    
    def _fetch_real_aws_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool) -> Optional[Dict]:
        """Fetch real pricing from AWS Pricing API"""
        
        if not self.pricing_client:
            raise Exception("AWS pricing client not available")
        
        # Map our engine names to AWS service codes
        engine_mapping = {
            'postgres': 'PostgreSQL',
            'mysql': 'MySQL',
            'aurora-postgresql': 'Aurora PostgreSQL',
            'aurora-mysql': 'Aurora MySQL',
            'oracle-ee': 'Oracle',
            'oracle-se': 'Oracle',
            'sql-server': 'SQL Server'
        }
        
        try:
            # Get RDS pricing
            response = self.pricing_client.get_products(
                ServiceCode='AmazonRDS',
                Filters=[
                    {
                        'Type': 'TERM_MATCH',
                        'Field': 'location',
                        'Value': self._get_aws_region_name(region)
                    },
                    {
                        'Type': 'TERM_MATCH',
                        'Field': 'instanceType',
                        'Value': instance_class
                    },
                    {
                        'Type': 'TERM_MATCH',
                        'Field': 'databaseEngine',
                        'Value': engine_mapping.get(engine, 'PostgreSQL')
                    },
                    {
                        'Type': 'TERM_MATCH',
                        'Field': 'deploymentOption',
                        'Value': 'Multi-AZ' if multi_az else 'Single-AZ'
                    }
                ],
                MaxResults=10
            )
            
            # Parse pricing data
            if response.get('PriceList'):
                price_data = json.loads(response['PriceList'][0])
                
                # Extract hourly rate
                terms = price_data.get('terms', {})
                on_demand = terms.get('OnDemand', {})
                
                for term_key, term_data in on_demand.items():
                    price_dimensions = term_data.get('priceDimensions', {})
                    for dim_key, dim_data in price_dimensions.items():
                        price_per_unit = float(dim_data.get('pricePerUnit', {}).get('USD', '0'))
                        
                        if price_per_unit > 0:
                            return {
                                'hourly': price_per_unit,
                                'storage_gb': 0.115,  # Default storage pricing
                                'iops_gb': 0.10,
                                'io_request': 0.20,
                                'is_aurora': 'aurora' in engine,
                                'multi_az': multi_az,
                                'source': 'AWS Pricing API'
                            }
            
            return None
            
        except Exception as e:
            print(f"Error in AWS Pricing API call: {e}")
            return None
    
    def _get_aws_region_name(self, region_code: str) -> str:
        """Convert region code to AWS region name"""
        region_mapping = {
            'us-east-1': 'US East (N. Virginia)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'Europe (Ireland)',
            'ap-southeast-1': 'Asia Pacific (Singapore)',
            'ap-northeast-1': 'Asia Pacific (Tokyo)'
        }
        return region_mapping.get(region_code, 'US East (N. Virginia)')
    
    def _get_fallback_pricing(self, region: str, engine: str, instance_class: str, multi_az: bool) -> Dict:
        """Fallback pricing when API is unavailable"""
        
        # Your existing hardcoded pricing as fallback
        pricing_data = {
            'us-east-1': {
                'postgres': {
                    'db.t3.micro': {'hourly': 0.0255, 'hourly_multi_az': 0.051, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.small': {'hourly': 0.051, 'hourly_multi_az': 0.102, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.medium': {'hourly': 0.102, 'hourly_multi_az': 0.204, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.t3.large': {'hourly': 0.204, 'hourly_multi_az': 0.408, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.large': {'hourly': 0.24, 'hourly_multi_az': 0.48, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.xlarge': {'hourly': 0.48, 'hourly_multi_az': 0.96, 'storage_gb': 0.115, 'iops_gb': 0.10},
                    'db.r5.2xlarge': {'hourly': 0.96, 'hourly_multi_az': 1.92, 'storage_gb': 0.115, 'iops_gb': 0.10},
                }
            }
        }
        
        engine_pricing = pricing_data.get(region, {}).get(engine, {})
        instance_pricing = engine_pricing.get(instance_class, {
            'hourly': 0.5, 
            'hourly_multi_az': 1.0, 
            'storage_gb': 0.115, 
            'iops_gb': 0.10
        })
        
        hourly_cost = instance_pricing['hourly_multi_az'] if multi_az else instance_pricing['hourly']
        
        return {
            'hourly': hourly_cost,
            'storage_gb': instance_pricing['storage_gb'],
            'iops_gb': instance_pricing.get('iops_gb', 0.10),
            'io_request': 0.20,
            'is_aurora': 'aurora' in engine,
            'multi_az': multi_az,
            'source': 'Fallback Pricing'
        }


# REAL Claude AI Implementation
class RealClaudeAIAnalyzer:
    """Real Claude AI integration for migration insights"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        
        if api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Anthropic client: {e}")
    
    async def generate_real_ai_insights(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Generate real AI insights using Claude"""
        
        if not self.client:
            return {
                'error': 'Claude AI not available - using fallback insights',
                'fallback_insights': self._get_fallback_insights(cost_analysis, migration_params)
            }
        
        try:
            # Prepare context for Claude
            context = self._prepare_migration_context(cost_analysis, migration_params)
            
            # Create the prompt
            prompt = f"""
            You are an AWS migration expert analyzing a database migration project. 
            Please provide detailed insights and recommendations based on the following migration analysis:

            {context}

            Please provide insights in the following categories:
            1. Cost Analysis & Optimization
            2. Risk Assessment
            3. Migration Strategy Recommendations
            4. Timeline Feasibility
            5. Technical Considerations
            6. Post-Migration Optimization

            Format your response as a structured analysis with specific, actionable recommendations.
            """
            
            # Call Claude API
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse Claude's response
            ai_response = message.content[0].text
            
            # Structure the response
            structured_insights = self._structure_ai_response(ai_response)
            structured_insights['source'] = 'Claude AI'
            structured_insights['model'] = 'claude-3-sonnet-20240229'
            
            return structured_insights
            
        except Exception as e:
            print(f"Error calling Claude AI: {e}")
            return {
                'error': f'Claude AI call failed: {str(e)}',
                'fallback_insights': self._get_fallback_insights(cost_analysis, migration_params)
            }
    
    def _prepare_migration_context(self, cost_analysis: Dict, migration_params: Dict) -> str:
        """Prepare context for Claude AI"""
        
        context = f"""
        MIGRATION PROJECT OVERVIEW:
        - Source Database: {migration_params.get('source_engine', 'Unknown')}
        - Target Database: {migration_params.get('target_engine', 'Unknown')}
        - Data Size: {migration_params.get('data_size_gb', 0):,} GB
        - Timeline: {migration_params.get('migration_timeline_weeks', 0)} weeks
        - Team Size: {migration_params.get('team_size', 0)} members
        - Budget: ${migration_params.get('migration_budget', 0):,}

        COST ANALYSIS:
        - Monthly AWS Cost: ${cost_analysis.get('monthly_aws_cost', 0):,.0f}
        - Annual AWS Cost: ${cost_analysis.get('annual_aws_cost', 0):,.0f}
        - Migration Investment: ${cost_analysis.get('migration_costs', {}).get('total', 0):,.0f}

        ENVIRONMENTS:
        - Number of Environments: {len(cost_analysis.get('environment_costs', {}))}
        - Environment Details: {list(cost_analysis.get('environment_costs', {}).keys())}

        MIGRATION COMPLEXITY:
        - Applications Connected: {migration_params.get('num_applications', 0)}
        - Stored Procedures: {migration_params.get('num_stored_procedures', 0)}
        - Team Expertise: {migration_params.get('team_expertise', 'Unknown')}
        """
        
        return context
    
    def _structure_ai_response(self, ai_response: str) -> Dict:
        """Structure Claude's response into categories"""
        
        # Simple parsing - in production, you might want more sophisticated parsing
        sections = ai_response.split('\n\n')
        
        structured = {
            'cost_analysis': '',
            'risk_assessment': '',
            'migration_strategy': '',
            'timeline_analysis': '',
            'technical_recommendations': '',
            'optimization_opportunities': '',
            'full_response': ai_response
        }
        
        # Try to extract sections (basic implementation)
        current_section = 'general_insights'
        for section in sections:
            section_lower = section.lower()
            
            if 'cost' in section_lower:
                structured['cost_analysis'] += section + '\n\n'
            elif 'risk' in section_lower:
                structured['risk_assessment'] += section + '\n\n'
            elif 'strategy' in section_lower or 'migration' in section_lower:
                structured['migration_strategy'] += section + '\n\n'
            elif 'timeline' in section_lower or 'schedule' in section_lower:
                structured['timeline_analysis'] += section + '\n\n'
            elif 'technical' in section_lower:
                structured['technical_recommendations'] += section + '\n\n'
            elif 'optimization' in section_lower:
                structured['optimization_opportunities'] += section + '\n\n'
        
        return structured
    
    def _get_fallback_insights(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Fallback insights when AI is unavailable"""
        
        monthly_cost = cost_analysis.get('monthly_aws_cost', 0)
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        data_size = migration_params.get('data_size_gb', 0)
        
        return {
            'cost_analysis': f"Monthly AWS cost of ${monthly_cost:,.0f} appears reasonable for a migration of this scale. Consider Reserved Instances for 30-40% savings.",
            'risk_assessment': f"Migration timeline of {timeline_weeks} weeks is adequate for {data_size:,} GB of data. Key risks include application compatibility and data validation.",
            'migration_strategy': "Recommend phased approach starting with non-production environments. Use AWS DMS for initial sync with minimal downtime cutover.",
            'timeline_analysis': f"Timeline allows for proper testing and validation. Allocate 30% of time for testing and rollback planning.",
            'technical_recommendations': "Implement comprehensive monitoring, backup strategies, and performance baseline establishment before migration.",
            'optimization_opportunities': "Post-migration optimization opportunities include right-sizing instances, implementing read replicas, and storage optimization.",
            'source': 'Fallback Insights'
        }


class DatabaseClusterConfiguration:
    """Enhanced database cluster configuration with Writer/Reader support"""
    
    @staticmethod
    def calculate_optimal_readers(environment_type: str, workload_pattern: str, connections: int) -> int:
        """Calculate optimal number of read replicas"""
        
        base_readers = {
            'production': 2,
            'staging': 1,
            'testing': 1,
            'development': 0
        }
        
        # Adjust based on workload pattern
        workload_multipliers = {
            'read_heavy': 1.5,
            'balanced': 1.0,
            'write_heavy': 0.5,
            'analytics': 2.0
        }
        
        # Adjust based on connection count
        if connections > 1000:
            connection_factor = 1.5
        elif connections > 500:
            connection_factor = 1.2
        else:
            connection_factor = 1.0
        
        optimal_readers = int(
            base_readers.get(environment_type, 1) * 
            workload_multipliers.get(workload_pattern, 1.0) * 
            connection_factor
        )
        
        return max(0, min(optimal_readers, 5))  # Cap at 5 readers
    
    @staticmethod
    def recommend_reader_instance_size(writer_instance: str, read_ratio: float = 0.7) -> str:
        """Recommend reader instance size based on writer and read ratio"""
        
        # Instance sizing hierarchy
        instance_hierarchy = [
            'db.t3.micro', 'db.t3.small', 'db.t3.medium', 'db.t3.large', 'db.t3.xlarge',
            'db.r5.large', 'db.r5.xlarge', 'db.r5.2xlarge', 'db.r5.4xlarge', 'db.r5.8xlarge'
        ]
        
        try:
            writer_index = instance_hierarchy.index(writer_instance)
            
            # For heavy read workloads, readers might need to be same size or larger
            if read_ratio > 0.8:
                reader_index = writer_index
            elif read_ratio > 0.5:
                reader_index = max(0, writer_index - 1)
            else:
                reader_index = max(0, writer_index - 2)
            
            return instance_hierarchy[reader_index]
            
        except ValueError:
            # If writer instance not in hierarchy, default to r5.large
            return 'db.r5.large'
        
# Updated Migration Analyzer with Real APIs
class RealMigrationAnalyzer:
    """Migration analyzer with real AWS pricing and Claude AI"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.pricing_api = RealAWSPricingAPI()
        self.ai_analyzer = RealClaudeAIAnalyzer(anthropic_api_key)
        self.anthropic_api_key = anthropic_api_key
    
    def calculate_instance_recommendations(self, environment_specs: Dict) -> Dict:
        """Calculate AWS instance recommendations using real pricing"""
        
        recommendations = {}
        
        for env_name, specs in environment_specs.items():
            cpu_cores = specs['cpu_cores']
            ram_gb = specs['ram_gb']
            storage_gb = specs['storage_gb']
            
            # Determine environment type
            environment_type = self._categorize_environment(env_name)
            
            # Calculate instance class
            instance_class = self._calculate_instance_class(cpu_cores, ram_gb, environment_type)
            
            # Multi-AZ recommendation
            multi_az = environment_type in ['production', 'staging']
            
            recommendations[env_name] = {
                'environment_type': environment_type,
                'instance_class': instance_class,
                'cpu_cores': cpu_cores,
                'ram_gb': ram_gb,
                'storage_gb': storage_gb,
                'multi_az': multi_az,
                'daily_usage_hours': specs.get('daily_usage_hours', 24),
                'peak_connections': specs.get('peak_connections', 100)
            }
        
        return recommendations
    
    def calculate_migration_costs(self, recommendations: Dict, migration_params: Dict) -> Dict:
        """Calculate migration costs using REAL AWS pricing"""
        
        region = migration_params.get('region', 'us-east-1')
        target_engine = migration_params.get('target_engine', 'postgres')
        
        total_monthly_cost = 0
        environment_costs = {}
        
        for env_name, rec in recommendations.items():
            # Use REAL pricing API
            env_costs = self._calculate_environment_cost_real(env_name, rec, region, target_engine)
            environment_costs[env_name] = env_costs
            total_monthly_cost += env_costs['total_monthly']
        
        # Migration service costs (same as before)
        data_size_gb = migration_params.get('data_size_gb', 1000)
        migration_timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        # DMS costs
        dms_instance_cost = 0.2 * 24 * 7 * migration_timeline_weeks
        
        # Data transfer costs
        transfer_costs = self._calculate_transfer_costs(data_size_gb, migration_params)
        
        # Professional services
        ps_cost = migration_timeline_weeks * 8000
        
        migration_costs = {
            'dms_instance': dms_instance_cost,
            'data_transfer': transfer_costs.get('total', data_size_gb * 0.09),
            'professional_services': ps_cost,
            'contingency': 0,
            'total': 0
        }
        
        base_cost = migration_costs['dms_instance'] + migration_costs['data_transfer'] + migration_costs['professional_services']
        migration_costs['contingency'] = base_cost * 0.2
        migration_costs['total'] = base_cost + migration_costs['contingency']
        
        return {
            'monthly_aws_cost': total_monthly_cost,
            'annual_aws_cost': total_monthly_cost * 12,
            'environment_costs': environment_costs,
            'migration_costs': migration_costs,
            'transfer_costs': transfer_costs
        }
    
    def _calculate_environment_cost_real(self, env_name: str, rec: Dict, region: str, target_engine: str) -> Dict:
        """Calculate environment cost using REAL AWS pricing"""
        
        # Get REAL pricing from AWS API
        pricing = self.pricing_api.get_rds_pricing(
            region, target_engine, rec['instance_class'], rec['multi_az']
        )
        
        # Calculate monthly hours
        daily_hours = rec['daily_usage_hours']
        monthly_hours = daily_hours * 30
        
        # Instance cost using REAL pricing
        instance_cost = pricing['hourly'] * monthly_hours
        
        # Storage cost using REAL pricing
        storage_cost = rec['storage_gb'] * pricing['storage_gb']
        
        # Backup cost (estimate 20% of storage)
        backup_cost = storage_cost * 0.2
        
        # Total monthly cost
        total_monthly = instance_cost + storage_cost + backup_cost
        
        return {
            'instance_cost': instance_cost,
            'storage_cost': storage_cost,
            'backup_cost': backup_cost,
            'total_monthly': total_monthly,
            'pricing_source': pricing.get('source', 'Unknown')
        }
    
    async def generate_real_ai_insights(self, cost_analysis: Dict, migration_params: Dict) -> Dict:
        """Generate REAL AI insights using Claude"""
        return await self.ai_analyzer.generate_real_ai_insights(cost_analysis, migration_params)
    
    # Include other methods from original MigrationAnalyzer...
    def _categorize_environment(self, env_name: str) -> str:
        """Categorize environment type from name"""
        env_lower = env_name.lower()
        if any(term in env_lower for term in ['prod', 'production', 'prd']):
            return 'production'
        elif any(term in env_lower for term in ['stag', 'staging', 'preprod']):
            return 'staging'
        elif any(term in env_lower for term in ['qa', 'test', 'uat', 'sqa']):
            return 'testing'
        elif any(term in env_lower for term in ['dev', 'development', 'sandbox']):
            return 'development'
        return 'production'
    
    def _calculate_instance_class(self, cpu_cores: int, ram_gb: int, env_type: str) -> str:
        """Calculate appropriate instance class"""
        
        if cpu_cores <= 2 and ram_gb <= 8:
            instance_class = 'db.t3.medium'
        elif cpu_cores <= 4 and ram_gb <= 16:
            instance_class = 'db.t3.large'
        elif cpu_cores <= 8 and ram_gb <= 32:
            instance_class = 'db.r5.large'
        elif cpu_cores <= 16 and ram_gb <= 64:
            instance_class = 'db.r5.xlarge'
        elif cpu_cores <= 32 and ram_gb <= 128:
            instance_class = 'db.r5.2xlarge'
        elif cpu_cores <= 64 and ram_gb <= 256:
            instance_class = 'db.r5.4xlarge'
        else:
            instance_class = 'db.r5.8xlarge'
        
        # Environment-specific adjustments
        if env_type == 'development' and 'r5' in instance_class:
            downsized = {
                'db.r5.8xlarge': 'db.r5.4xlarge',
                'db.r5.4xlarge': 'db.r5.2xlarge',
                'db.r5.2xlarge': 'db.r5.xlarge',
                'db.r5.xlarge': 'db.r5.large',
                'db.r5.large': 'db.t3.large'
            }
            instance_class = downsized.get(instance_class, instance_class)
        
        return instance_class
    
    def _calculate_transfer_costs(self, data_size_gb: int, migration_params: Dict) -> Dict:
        """Calculate data transfer costs"""
        
        use_direct_connect = migration_params.get('use_direct_connect', False)
        
        internet_cost = data_size_gb * 0.09
        
        if use_direct_connect:
            dx_cost = data_size_gb * 0.02
        else:
            dx_cost = internet_cost
        
        return {
            'internet': internet_cost,
            'direct_connect': dx_cost,
            'total': min(internet_cost, dx_cost)
        }


# Updated analysis function to use real APIs
def run_real_migration_analysis():
    """Run migration analysis with REAL AWS pricing and Claude AI"""
    
    try:
        # Initialize REAL analyzer
        anthropic_api_key = st.session_state.migration_params.get('anthropic_api_key')
        analyzer = RealMigrationAnalyzer(anthropic_api_key)
        
        # Step 1: Calculate recommendations with REAL AWS pricing
        st.write("📊 Calculating instance recommendations with live AWS pricing...")
        recommendations = analyzer.calculate_instance_recommendations(st.session_state.environment_specs)
        st.session_state.recommendations = recommendations
        
        # Step 2: Calculate costs with REAL AWS pricing
        st.write("💰 Analyzing costs with real-time AWS pricing...")
        cost_analysis = analyzer.calculate_migration_costs(recommendations, st.session_state.migration_params)
        
        # Check pricing sources
        pricing_sources = set()
        for env_costs in cost_analysis['environment_costs'].values():
            pricing_sources.add(env_costs.get('pricing_source', 'Unknown'))
        
        if 'AWS Pricing API' in pricing_sources:
            st.success("✅ Using real-time AWS pricing data")
        else:
            st.warning("⚠️ Using fallback pricing data (AWS API unavailable)")
        
        # Add reconciliation metadata
        if use_reconciliation:
            cost_analysis['reconciliation_applied'] = True
            cost_analysis['reconciliation_settings'] = reconciliation_settings
        
        
        st.session_state.analysis_results = cost_analysis
        
        # Step 3: Risk assessment
        st.write("⚠️ Assessing risks...")
        risk_assessment = create_default_risk_assessment()
        st.session_state.risk_assessment = risk_assessment
        
        # Step 4: REAL AI insights
        if anthropic_api_key:
            st.write("🤖 Generating AI insights with Claude...")
            try:
                ai_insights = analyzer.generate_ai_insights_sync(cost_analysis, st.session_state.migration_params)
                st.session_state.ai_insights = ai_insights
                
                if ai_insights.get('source') == 'Claude AI':
                    st.success("✅ AI insights generated by Claude AI")
                else:
                    st.warning("⚠️ Using fallback AI insights (Claude API unavailable)")
                    
            except Exception as e:
                st.warning(f"Claude AI call failed: {str(e)}")
                st.session_state.ai_insights = {'error': str(e)}
        else:
            st.info("ℹ️ Provide Anthropic API key for Claude AI insights")
        
        st.success("✅ Analysis complete with real-time data!")
       
        # Suggest reconciliation if not already enabled
        if not use_reconciliation:
            st.info("💡 Visit the 'Cost Reconciliation' section to validate these costs against AWS Cost Analysis")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")


# Installation requirements for real APIs
def show_real_api_setup():
    """Show setup instructions for real API connections"""
    
    st.markdown("### 🔧 Real API Setup Instructions")
    
    st.markdown("""
    To enable real AWS pricing and Claude AI integration, you need:

    #### 1. AWS Pricing API
    ```bash
    pip install boto3
    ```
    
    **AWS Credentials Setup:**
    - Configure AWS credentials: `aws configure`
    - Or set environment variables:
      ```bash
      export AWS_ACCESS_KEY_ID=your_access_key
      export AWS_SECRET_ACCESS_KEY=your_secret_key
      ```
    
    #### 2. Claude AI Integration
    ```bash
    pip install anthropic
    ```
    
    **API Key Setup:**
    - Get API key from: https://console.anthropic.com/
    - Enter key in Migration Configuration section
    
    #### 3. Required Permissions
    **AWS IAM Policy:**
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "pricing:GetProducts",
                    "pricing:DescribeServices"
                ],
                "Resource": "*"
            }
        ]
    }
    ```
    """)
# ===========================
# NETWORK TRANSFER ANALYSIS MODULE
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json



class NetworkTransferAnalyzer:
    """Comprehensive network transfer analysis for AWS database migration"""
    
    def __init__(self):
        self.transfer_patterns = self._initialize_transfer_patterns()
        self.aws_regions_bandwidth = self._initialize_region_bandwidth()
        
    def _initialize_transfer_patterns(self) -> Dict:
        """Initialize supported network transfer patterns"""
        return {
            'internet_dms': {
                'name': 'Internet + DMS',
                'description': 'Standard internet connection with AWS Database Migration Service',
                'components': ['Internet Gateway', 'DMS Replication Instance'],
                'use_cases': ['Small to medium databases (<1TB)', 'Standard migrations', 'Cost-sensitive projects'],
                'pros': ['Low setup cost', 'Quick to implement', 'No additional infrastructure'],
                'cons': ['Variable bandwidth', 'Higher data transfer costs', 'Potential latency issues'],
                'security_level': 'Standard',
                'complexity': 'Low'
            },
            'dx_dms': {
                'name': 'Direct Connect + DMS',
                'description': 'Dedicated connection with AWS Database Migration Service',
                'components': ['AWS Direct Connect', 'DMS Replication Instance', 'Virtual Interface'],
                'use_cases': ['Large databases (>1TB)', 'Consistent bandwidth needs', 'Ongoing hybrid connectivity'],
                'pros': ['Predictable bandwidth', 'Lower data transfer costs', 'Reduced latency'],
                'cons': ['Higher setup cost', 'Longer setup time', 'Requires dedicated circuit'],
                'security_level': 'High',
                'complexity': 'Medium'
            },
            'dx_datasync_vpc': {
                'name': 'Direct Connect + DataSync + VPC Endpoints',
                'description': 'Dedicated connection with DataSync using VPC endpoints for private connectivity',
                'components': ['AWS Direct Connect', 'DataSync Agent', 'VPC Endpoints', 'S3 Gateway Endpoint'],
                'use_cases': ['File-based data transfer', 'Object storage migration', 'High security requirements'],
                'pros': ['Private connectivity', 'Optimized for file transfer', 'No internet routing'],
                'cons': ['Complex setup', 'Additional VPC endpoint costs', 'Limited to file-based transfers'],
                'security_level': 'Very High',
                'complexity': 'High'
            },
            'vpn_dms': {
                'name': 'VPN + DMS',
                'description': 'Site-to-site VPN with AWS Database Migration Service',
                'components': ['VPN Gateway', 'Customer Gateway', 'DMS Replication Instance'],
                'use_cases': ['Medium databases', 'Secure connectivity required', 'Temporary migration setup'],
                'pros': ['Secure connection', 'Quick setup', 'Cost-effective'],
                'cons': ['Internet-dependent', 'Bandwidth limitations', 'Potential latency'],
                'security_level': 'High',
                'complexity': 'Medium'
            },
            'hybrid_snowball_dms': {
                'name': 'Snowball + DMS Hybrid',
                'description': 'Initial bulk transfer via Snowball, ongoing sync with DMS',
                'components': ['AWS Snowball', 'DMS Replication Instance', 'S3 Bucket'],
                'use_cases': ['Very large databases (>10TB)', 'Limited bandwidth', 'Minimal downtime required'],
                'pros': ['Fast initial transfer', 'Minimal bandwidth usage', 'Reduced downtime'],
                'cons': ['Complex orchestration', 'Physical device handling', 'Higher coordination effort'],
                'security_level': 'High',
                'complexity': 'Very High'
            },
            'multipath_redundant': {
                'name': 'Multi-path Redundant Transfer',
                'description': 'Redundant connections using both Direct Connect and VPN',
                'components': ['AWS Direct Connect', 'VPN Gateway', 'DMS Replication Instance', 'Route Tables'],
                'use_cases': ['Mission-critical migrations', 'Zero-tolerance for failures', 'High availability requirements'],
                'pros': ['Maximum reliability', 'Automatic failover', 'Redundant paths'],
                'cons': ['Highest cost', 'Complex configuration', 'Over-engineering for most use cases'],
                'security_level': 'Very High',
                'complexity': 'Very High'
            }
        }
    
    def _initialize_region_bandwidth(self) -> Dict:
        """Initialize typical bandwidth capabilities by region"""
        return {
            'us-east-1': {'max_dx_gbps': 100, 'typical_internet_mbps': 1000},
            'us-west-2': {'max_dx_gbps': 100, 'typical_internet_mbps': 1000},
            'eu-west-1': {'max_dx_gbps': 100, 'typical_internet_mbps': 800},
            'ap-southeast-1': {'max_dx_gbps': 50, 'typical_internet_mbps': 500},
            'ap-northeast-1': {'max_dx_gbps': 100, 'typical_internet_mbps': 800}
        }
    
    def calculate_transfer_analysis(self, migration_params: Dict) -> Dict:
        """Calculate comprehensive transfer analysis for all patterns"""
        
        data_size_gb = migration_params.get('data_size_gb', 1000)
        region = migration_params.get('region', 'us-east-1')
        available_bandwidth_mbps = migration_params.get('bandwidth_mbps', 1000)
        security_requirements = migration_params.get('security_requirements', 'standard')
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        budget_constraints = migration_params.get('budget_constraints', 'medium')
        
        results = {}
        
        for pattern_id, pattern_info in self.transfer_patterns.items():
            results[pattern_id] = self._calculate_pattern_metrics(
                pattern_id, pattern_info, data_size_gb, region, 
                available_bandwidth_mbps, security_requirements, timeline_weeks
            )
        
        # Add recommendation engine
        results['recommendations'] = self._generate_recommendations(
            results, migration_params
        )
        
        return results
    
    def _calculate_pattern_metrics(self, pattern_id: str, pattern_info: Dict, 
                                 data_size_gb: int, region: str, bandwidth_mbps: int,
                                 security_req: str, timeline_weeks: int) -> Dict:
        """Calculate metrics for a specific transfer pattern"""
        
        # Base calculations
        data_size_tb = data_size_gb / 1024
        
        # Pattern-specific calculations
        if pattern_id == 'internet_dms':
            return self._calc_internet_dms(data_size_gb, bandwidth_mbps, region)
        elif pattern_id == 'dx_dms':
            return self._calc_dx_dms(data_size_gb, bandwidth_mbps, region)
        elif pattern_id == 'dx_datasync_vpc':
            return self._calc_dx_datasync_vpc(data_size_gb, bandwidth_mbps, region)
        elif pattern_id == 'vpn_dms':
            return self._calc_vpn_dms(data_size_gb, bandwidth_mbps, region)
        elif pattern_id == 'hybrid_snowball_dms':
            return self._calc_hybrid_snowball(data_size_gb, bandwidth_mbps, region)
        elif pattern_id == 'multipath_redundant':
            return self._calc_multipath_redundant(data_size_gb, bandwidth_mbps, region)
        
        return {}
    
    def _calc_internet_dms(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate Internet + DMS pattern metrics"""
        
        # Transfer time calculation (with 70% efficiency factor for internet)
        effective_bandwidth = bandwidth_mbps * 0.7
        transfer_time_hours = (data_size_gb * 8 * 1024) / (effective_bandwidth * 3600)  # Convert to hours
        
        # Cost calculations
        data_transfer_cost = data_size_gb * 0.09  # $0.09 per GB for internet transfer
        dms_instance_cost = 0.34 * (transfer_time_hours / 24)  # t3.large DMS instance per day
        setup_cost = 2000  # Basic setup costs
        
        total_cost = data_transfer_cost + dms_instance_cost + setup_cost
        
        return {
            'transfer_time_hours': transfer_time_hours,
            'transfer_time_days': transfer_time_hours / 24,
            'data_transfer_cost': data_transfer_cost,
            'infrastructure_cost': dms_instance_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 70,  # Percentage
            'reliability_score': 75,
            'security_score': 60,
            'complexity_score': 20
        }
    
    def _calc_dx_dms(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate Direct Connect + DMS pattern metrics"""
        
        # Transfer time calculation (with 95% efficiency for DX)
        effective_bandwidth = bandwidth_mbps * 0.95
        transfer_time_hours = (data_size_gb * 8 * 1024) / (effective_bandwidth * 3600)
        
        # Cost calculations
        data_transfer_cost = data_size_gb * 0.02  # $0.02 per GB for DX transfer
        dx_port_cost = 500 * (transfer_time_hours / (24 * 30))  # $500/month for 1Gbps port
        dms_instance_cost = 0.34 * (transfer_time_hours / 24)
        setup_cost = 8000  # DX setup costs
        
        total_cost = data_transfer_cost + dx_port_cost + dms_instance_cost + setup_cost
        
        return {
            'transfer_time_hours': transfer_time_hours,
            'transfer_time_days': transfer_time_hours / 24,
            'data_transfer_cost': data_transfer_cost,
            'infrastructure_cost': dx_port_cost + dms_instance_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 95,
            'reliability_score': 95,
            'security_score': 85,
            'complexity_score': 60
        }
    
    def _calc_dx_datasync_vpc(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate DX + DataSync + VPC Endpoints pattern metrics"""
        
        # Transfer time calculation (with 90% efficiency)
        effective_bandwidth = bandwidth_mbps * 0.90
        transfer_time_hours = (data_size_gb * 8 * 1024) / (effective_bandwidth * 3600)
        
        # Cost calculations
        data_transfer_cost = data_size_gb * 0.0125  # DataSync per GB
        dx_port_cost = 500 * (transfer_time_hours / (24 * 30))
        vpc_endpoint_cost = 22.5 * (transfer_time_hours / (24 * 30))  # $22.50/month per endpoint
        datasync_agent_cost = 0.048 * (transfer_time_hours / 24)  # m5.large for agent
        setup_cost = 12000  # Higher setup due to VPC endpoints configuration
        
        total_cost = data_transfer_cost + dx_port_cost + vpc_endpoint_cost + datasync_agent_cost + setup_cost
        
        return {
            'transfer_time_hours': transfer_time_hours,
            'transfer_time_days': transfer_time_hours / 24,
            'data_transfer_cost': data_transfer_cost,
            'infrastructure_cost': dx_port_cost + vpc_endpoint_cost + datasync_agent_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 90,
            'reliability_score': 90,
            'security_score': 95,
            'complexity_score': 85
        }
    
    def _calc_vpn_dms(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate VPN + DMS pattern metrics"""
        
        # Transfer time calculation (with 80% efficiency for VPN)
        effective_bandwidth = min(bandwidth_mbps * 0.8, 1250)  # VPN Gateway limit ~1.25 Gbps
        transfer_time_hours = (data_size_gb * 8 * 1024) / (effective_bandwidth * 3600)
        
        # Cost calculations
        data_transfer_cost = data_size_gb * 0.09  # Same as internet
        vpn_gateway_cost = 36 * (transfer_time_hours / (24 * 30))  # $36/month
        dms_instance_cost = 0.34 * (transfer_time_hours / 24)
        setup_cost = 3000  # VPN setup costs
        
        total_cost = data_transfer_cost + vpn_gateway_cost + dms_instance_cost + setup_cost
        
        return {
            'transfer_time_hours': transfer_time_hours,
            'transfer_time_days': transfer_time_hours / 24,
            'data_transfer_cost': data_transfer_cost,
            'infrastructure_cost': vpn_gateway_cost + dms_instance_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 80,
            'reliability_score': 85,
            'security_score': 90,
            'complexity_score': 50
        }
    
    def _calc_hybrid_snowball(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate Snowball + DMS hybrid pattern metrics"""
        
        # Assume 80% via Snowball, 20% via DMS for ongoing sync
        snowball_data_gb = data_size_gb * 0.8
        dms_data_gb = data_size_gb * 0.2
        
        # Snowball calculations
        snowball_devices = max(1, int(snowball_data_gb / (80 * 1024)))  # 80TB per device
        snowball_cost = snowball_devices * 250  # $250 per device
        snowball_shipping_days = 5  # Average shipping time
        
        # DMS for ongoing sync
        effective_bandwidth = bandwidth_mbps * 0.7
        dms_transfer_hours = (dms_data_gb * 8 * 1024) / (effective_bandwidth * 3600)
        dms_cost = dms_data_gb * 0.09 + 0.34 * (dms_transfer_hours / 24)
        
        setup_cost = 15000  # Complex orchestration setup
        
        total_transfer_days = snowball_shipping_days + (dms_transfer_hours / 24)
        total_cost = snowball_cost + dms_cost + setup_cost
        
        return {
            'transfer_time_hours': total_transfer_days * 24,
            'transfer_time_days': total_transfer_days,
            'data_transfer_cost': snowball_cost,
            'infrastructure_cost': dms_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 95,  # High for Snowball portion
            'reliability_score': 90,
            'security_score': 85,
            'complexity_score': 95
        }
    
    def _calc_multipath_redundant(self, data_size_gb: int, bandwidth_mbps: int, region: str) -> Dict:
        """Calculate multi-path redundant pattern metrics"""
        
        # Use both DX and VPN, with load balancing
        effective_bandwidth = bandwidth_mbps * 1.2  # 20% boost from dual paths
        transfer_time_hours = (data_size_gb * 8 * 1024) / (effective_bandwidth * 3600)
        
        # Cost calculations (combination of DX and VPN costs)
        dx_costs = self._calc_dx_dms(data_size_gb, bandwidth_mbps, region)
        vpn_costs = self._calc_vpn_dms(data_size_gb, bandwidth_mbps, region)
        
        # Take higher infrastructure costs + additional setup
        data_transfer_cost = min(dx_costs['data_transfer_cost'], vpn_costs['data_transfer_cost'])
        infrastructure_cost = dx_costs['infrastructure_cost'] + vpn_costs['infrastructure_cost']
        setup_cost = 20000  # Complex dual-path setup
        
        total_cost = data_transfer_cost + infrastructure_cost + setup_cost
        
        return {
            'transfer_time_hours': transfer_time_hours,
            'transfer_time_days': transfer_time_hours / 24,
            'data_transfer_cost': data_transfer_cost,
            'infrastructure_cost': infrastructure_cost,
            'setup_cost': setup_cost,
            'total_cost': total_cost,
            'bandwidth_utilization': 98,
            'reliability_score': 99,
            'security_score': 95,
            'complexity_score': 100
        }
    
    def _generate_recommendations(self, results: Dict, migration_params: Dict) -> Dict:
        """Generate AI-style recommendations for network transfer patterns"""
        
        data_size_gb = migration_params.get('data_size_gb', 1000)
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        budget_level = migration_params.get('budget_constraints', 'medium')
        security_req = migration_params.get('security_requirements', 'standard')
        
        # Score each pattern
        pattern_scores = {}
        
        for pattern_id, metrics in results.items():
            if pattern_id == 'recommendations':
                continue
                
            # Calculate composite score based on multiple factors
            cost_score = self._score_cost(metrics['total_cost'], budget_level)
            time_score = self._score_time(metrics['transfer_time_days'], timeline_weeks)
            security_score = metrics['security_score']
            reliability_score = metrics['reliability_score']
            complexity_penalty = 100 - metrics['complexity_score']
            
            # Weighted scoring
            composite_score = (
                cost_score * 0.25 +
                time_score * 0.25 +
                security_score * 0.20 +
                reliability_score * 0.20 +
                complexity_penalty * 0.10
            )
            
            pattern_scores[pattern_id] = {
                'composite_score': composite_score,
                'cost_score': cost_score,
                'time_score': time_score,
                'metrics': metrics
            }
        
        # Sort by composite score
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        # Generate recommendations
        recommendations = {
            'primary_recommendation': self._get_primary_recommendation(sorted_patterns, migration_params),
            'alternative_options': self._get_alternative_options(sorted_patterns[:3], migration_params),
            'cost_optimization': self._get_cost_optimization_tips(sorted_patterns, migration_params),
            'risk_considerations': self._get_risk_considerations(sorted_patterns, migration_params),
            'timeline_impact': self._get_timeline_impact(sorted_patterns, migration_params)
        }
        
        return recommendations
    
    def _score_cost(self, total_cost: float, budget_level: str) -> float:
        """Score based on cost relative to budget constraints"""
        budget_thresholds = {
            'low': 10000,
            'medium': 50000,
            'high': 200000
        }
        
        threshold = budget_thresholds.get(budget_level, 50000)
        
        if total_cost <= threshold * 0.5:
            return 100
        elif total_cost <= threshold:
            return 80
        elif total_cost <= threshold * 1.5:
            return 60
        elif total_cost <= threshold * 2:
            return 40
        else:
            return 20
    
    def _score_time(self, transfer_days: float, timeline_weeks: int) -> float:
        """Score based on time relative to migration timeline"""
        available_days = timeline_weeks * 7 * 0.3  # 30% of timeline for data transfer
        
        if transfer_days <= available_days * 0.3:
            return 100
        elif transfer_days <= available_days * 0.5:
            return 90
        elif transfer_days <= available_days * 0.7:
            return 75
        elif transfer_days <= available_days:
            return 60
        else:
            return 30
    
    def _get_primary_recommendation(self, sorted_patterns: List, migration_params: Dict) -> Dict:
        """Get primary recommendation with reasoning"""
        
        best_pattern_id, best_scores = sorted_patterns[0]
        pattern_info = self.transfer_patterns[best_pattern_id]
        
        reasoning = self._generate_recommendation_reasoning(
            best_pattern_id, best_scores, migration_params
        )
        
        return {
            'pattern_id': best_pattern_id,
            'pattern_name': pattern_info['name'],
            'description': pattern_info['description'],
            'score': best_scores['composite_score'],
            'reasoning': reasoning,
            'implementation_steps': self._get_implementation_steps(best_pattern_id),
            'estimated_timeline': self._get_implementation_timeline(best_pattern_id),
            'key_considerations': pattern_info['pros'][:3]
        }
    
    def _get_alternative_options(self, top_patterns: List, migration_params: Dict) -> List:
        """Get alternative options with brief explanations"""
        
        alternatives = []
        
        for pattern_id, scores in top_patterns[1:]:  # Skip the primary recommendation
            pattern_info = self.transfer_patterns[pattern_id]
            
            alternatives.append({
                'pattern_name': pattern_info['name'],
                'score': scores['composite_score'],
                'best_for': pattern_info['use_cases'][0],
                'trade_off': self._get_trade_off_analysis(pattern_id, top_patterns[0][0])
            })
        
        return alternatives
    
    def _generate_recommendation_reasoning(self, pattern_id: str, scores: Dict, migration_params: Dict) -> str:
        """Generate human-readable reasoning for recommendation"""
        
        data_size_gb = migration_params.get('data_size_gb', 1000)
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        reasoning_parts = []
        
        # Data size reasoning
        if data_size_gb < 500:
            reasoning_parts.append("Given the moderate data size")
        elif data_size_gb < 5000:
            reasoning_parts.append("Given the substantial data volume")
        else:
            reasoning_parts.append("Given the large-scale data migration requirements")
        
        # Cost reasoning
        if scores['cost_score'] > 80:
            reasoning_parts.append("this pattern offers excellent cost efficiency")
        elif scores['cost_score'] > 60:
            reasoning_parts.append("this pattern provides balanced cost considerations")
        else:
            reasoning_parts.append("while this pattern has higher costs, it provides superior capabilities")
        
        # Time reasoning
        if scores['time_score'] > 80:
            reasoning_parts.append("and fits well within your migration timeline")
        elif scores['time_score'] > 60:
            reasoning_parts.append("and aligns reasonably with your timeline constraints")
        else:
            reasoning_parts.append("though it may require timeline adjustments")
        
        return ". ".join(reasoning_parts) + "."
    
    def _get_implementation_steps(self, pattern_id: str) -> List[str]:
        """Get implementation steps for a pattern"""
        
        steps_map = {
            'internet_dms': [
                "Set up DMS replication instance in target AWS region",
                "Configure source and target database endpoints",
                "Create and configure replication task",
                "Test connectivity and run initial assessment",
                "Execute full load and ongoing replication"
            ],
            'dx_dms': [
                "Establish AWS Direct Connect connection",
                "Configure virtual interfaces and routing",
                "Set up DMS replication instance",
                "Configure database endpoints over DX",
                "Test connectivity and execute migration"
            ],
            'dx_datasync_vpc': [
                "Establish AWS Direct Connect connection",
                "Configure VPC endpoints for DataSync and S3",
                "Deploy DataSync agent in on-premises environment",
                "Create DataSync tasks for data transfer",
                "Monitor and validate data transfer completion"
            ],
            'vpn_dms': [
                "Set up VPN Gateway and Customer Gateway",
                "Establish site-to-site VPN connection",
                "Configure DMS replication instance",
                "Set up database endpoints over VPN",
                "Execute migration with monitoring"
            ],
            'hybrid_snowball_dms': [
                "Order and configure AWS Snowball devices",
                "Perform initial data extraction to Snowball",
                "Ship devices and import to S3",
                "Set up DMS for ongoing synchronization",
                "Coordinate final cutover timing"
            ],
            'multipath_redundant': [
                "Establish both Direct Connect and VPN connections",
                "Configure BGP routing with path preferences",
                "Set up redundant DMS instances",
                "Test failover scenarios",
                "Execute migration with active monitoring"
            ]
        }
        
        return steps_map.get(pattern_id, ["Pattern-specific steps not defined"])
    
    def _get_implementation_timeline(self, pattern_id: str) -> str:
        """Get estimated implementation timeline"""
        
        timeline_map = {
            'internet_dms': "1-2 weeks",
            'dx_dms': "4-6 weeks (including DX provisioning)",
            'dx_datasync_vpc': "6-8 weeks (complex VPC setup)",
            'vpn_dms': "2-3 weeks",
            'hybrid_snowball_dms': "8-12 weeks (includes shipping)",
            'multipath_redundant': "8-10 weeks (dual-path complexity)"
        }
        
        return timeline_map.get(pattern_id, "Timeline varies")
    
    def _get_trade_off_analysis(self, pattern_id: str, primary_pattern_id: str) -> str:
        """Get trade-off analysis between patterns"""
        
        # This would be more sophisticated in production
        trade_offs = {
            'internet_dms': "Lower setup cost but higher data transfer fees",
            'dx_dms': "Higher setup cost but better reliability and performance",
            'dx_datasync_vpc': "Maximum security but highest complexity",
            'vpn_dms': "Good security balance with moderate setup",
            'hybrid_snowball_dms': "Fastest for large data but complex coordination",
            'multipath_redundant': "Maximum reliability but highest cost"
        }
        
        return trade_offs.get(pattern_id, "Different cost/performance trade-offs")
    
    def _get_cost_optimization_tips(self, sorted_patterns: List, migration_params: Dict) -> List[str]:
        """Generate cost optimization recommendations"""
        
        tips = []
        
        # Analyze if Direct Connect is cost-effective
        dx_patterns = [p for p in sorted_patterns if 'dx' in p[0]]
        if dx_patterns and migration_params.get('data_size_gb', 1000) < 1000:
            tips.append("Consider internet-based transfer for smaller datasets to avoid Direct Connect setup costs")
        
        # Timeline optimization
        if migration_params.get('migration_timeline_weeks', 12) > 16:
            tips.append("Extended timeline allows for more cost-effective transfer methods")
        
        # Hybrid approach suggestion
        data_size = migration_params.get('data_size_gb', 1000)
        if data_size > 10000:
            tips.append("Consider hybrid Snowball approach for very large datasets to minimize bandwidth costs")
        
        tips.append("Implement data compression and deduplication to reduce transfer volumes")
        tips.append("Schedule transfers during off-peak hours to optimize bandwidth utilization")
        
        return tips
    
    def _get_risk_considerations(self, sorted_patterns: List, migration_params: Dict) -> List[str]:
        """Generate risk considerations"""
        
        risks = []
        
        primary_pattern = sorted_patterns[0][0]
        
        if 'internet' in primary_pattern:
            risks.append("Internet dependency may cause variable transfer speeds")
            risks.append("Consider backup connectivity options for critical migrations")
        
        if 'dx' in primary_pattern:
            risks.append("Direct Connect provisioning time may impact project timeline")
            risks.append("Single point of failure without redundant connectivity")
        
        if 'snowball' in primary_pattern:
            risks.append("Physical device logistics and coordination complexity")
            risks.append("Potential delays due to shipping and customs processes")
        
        if migration_params.get('data_size_gb', 1000) > 5000:
            risks.append("Large data volumes require careful bandwidth planning")
            risks.append("Consider phased approach to minimize business impact")
        
        return risks
    
    def _get_timeline_impact(self, sorted_patterns: List, migration_params: Dict) -> Dict:
        """Analyze timeline impact of different patterns"""
        
        timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
        
        fastest_pattern = min(sorted_patterns, key=lambda x: x[1]['metrics']['transfer_time_days'])
        slowest_pattern = max(sorted_patterns, key=lambda x: x[1]['metrics']['transfer_time_days'])
        
        return {
            'fastest_option': {
                'pattern': self.transfer_patterns[fastest_pattern[0]]['name'],
                'duration_days': fastest_pattern[1]['metrics']['transfer_time_days'],
                'timeline_utilization': f"{(fastest_pattern[1]['metrics']['transfer_time_days'] / (timeline_weeks * 7)) * 100:.1f}%"
            },
            'slowest_option': {
                'pattern': self.transfer_patterns[slowest_pattern[0]]['name'],
                'duration_days': slowest_pattern[1]['metrics']['transfer_time_days'],
                'timeline_utilization': f"{(slowest_pattern[1]['metrics']['transfer_time_days'] / (timeline_weeks * 7)) * 100:.1f}%"
            },
            'recommendation': "Plan for 20-30% buffer time beyond calculated transfer duration for testing and validation"
        }

# ===========================
# NETWORK VISUALIZATION FUNCTIONS
# ===========================

def create_network_comparison_chart(transfer_analysis: Dict) -> go.Figure:
    """Create comprehensive network pattern comparison chart"""
    
    patterns = []
    costs = []
    durations = []
    reliability = []
    security = []
    complexity = []
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        pattern_info = NetworkTransferAnalyzer().transfer_patterns[pattern_id]
        patterns.append(pattern_info['name'])
        costs.append(metrics['total_cost'])
        durations.append(metrics['transfer_time_days'])
        reliability.append(metrics['reliability_score'])
        security.append(metrics['security_score'])
        complexity.append(metrics['complexity_score'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Cost Comparison', 'Transfer Duration', 
                       'Reliability vs Security', 'Complexity Assessment'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Cost comparison
    fig.add_trace(
        go.Bar(x=patterns, y=costs, name='Total Cost ($)', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Duration comparison
    fig.add_trace(
        go.Bar(x=patterns, y=durations, name='Duration (days)', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Reliability vs Security scatter
    fig.add_trace(
        go.Scatter(x=reliability, y=security, mode='markers+text',
                  text=patterns, textposition="top center",
                  marker=dict(size=15, color=complexity, colorscale='Viridis',
                            showscale=True, colorbar=dict(title="Complexity Score")),
                  name='Reliability vs Security'),
        row=2, col=1
    )
    
    # Complexity radar-style (simplified as bar)
    fig.add_trace(
        go.Bar(x=patterns, y=complexity, name='Complexity Score', marker_color='coral'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Network Transfer Pattern Analysis",
        showlegend=False
    )
    
    # Update x-axis labels to be rotated
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(tickangle=45, row=i, col=j)
    
    return fig

def create_cost_duration_optimization_chart(transfer_analysis: Dict) -> go.Figure:
    """Create cost vs duration optimization chart"""
    
    patterns = []
    costs = []
    durations = []
    scores = []
    
    analyzer = NetworkTransferAnalyzer()
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        pattern_info = analyzer.transfer_patterns[pattern_id]
        patterns.append(pattern_info['name'])
        costs.append(metrics['total_cost'])
        durations.append(metrics['transfer_time_days'])
        
        # Calculate efficiency score (inverse of cost and duration)
        efficiency = 1000000 / (metrics['total_cost'] * metrics['transfer_time_days'])
        scores.append(efficiency)
    
    fig = go.Figure()
    
    # Create scatter plot
    fig.add_trace(go.Scatter(
        x=durations,
        y=costs,
        mode='markers+text',
        text=patterns,
        textposition="top center",
        marker=dict(
            size=[score/max(scores)*50 + 10 for score in scores],  # Size based on efficiency
            color=scores,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Efficiency Score")
        ),
        hovertemplate='<b>%{text}</b><br>' +
                      'Duration: %{x:.1f} days<br>' +
                      'Cost: $%{y:,.0f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Cost vs Duration Optimization Matrix',
        xaxis_title='Transfer Duration (days)',
        yaxis_title='Total Cost ($)',
        height=600,
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text="🎯 Ideal: Lower-left quadrant (Low cost, Low duration)",
                showarrow=False,
                font=dict(size=12, color="green"),
                bgcolor="lightgreen",
                bordercolor="green",
                borderwidth=1
            )
        ]
    )
    
    return fig

def create_network_architecture_diagram(selected_pattern: str) -> go.Figure:
    """Create network architecture diagram for selected pattern"""
    
    # Simplified architecture representation using plotly
    # In production, you might use more sophisticated diagramming tools
    
    analyzer = NetworkTransferAnalyzer()
    pattern_info = analyzer.transfer_patterns.get(selected_pattern, {})
    
    # Create a simple flow diagram
    fig = go.Figure()
    
    # Define positions for different components
    positions = {
        'on_premises': (1, 3),
        'internet': (3, 4),
        'dx': (3, 2),
        'vpn': (3, 3),
        'aws_services': (5, 3),
        'target_db': (7, 3)
    }
    
    # Add nodes
    for component, (x, y) in positions.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            text=[component.replace('_', ' ').title()],
            textposition="middle center",
            marker=dict(size=60, color='lightblue'),
            showlegend=False
        ))
    
    # Add connections based on pattern
    connections = []
    if 'internet' in selected_pattern:
        connections.extend([('on_premises', 'internet'), ('internet', 'aws_services')])
    if 'dx' in selected_pattern:
        connections.extend([('on_premises', 'dx'), ('dx', 'aws_services')])
    if 'vpn' in selected_pattern:
        connections.extend([('on_premises', 'vpn'), ('vpn', 'aws_services')])
    
    connections.append(('aws_services', 'target_db'))
    
    # Draw connections
    for start, end in connections:
        start_pos = positions[start]
        end_pos = positions[end]
        
        fig.add_trace(go.Scatter(
            x=[start_pos[0], end_pos[0]],
            y=[start_pos[1], end_pos[1]],
            mode='lines',
            line=dict(width=3, color='gray'),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f'Network Architecture: {pattern_info.get("name", selected_pattern)}',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=400,
        plot_bgcolor='white'
    )
    
    return fig

# ===========================
# STREAMLIT INTERFACE FUNCTIONS
# ===========================

def show_network_transfer_analysis():
    """Show network transfer analysis interface"""
    
    st.markdown("## 🌐 Network Transfer Analysis")
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Please complete Migration Configuration first.")
        return
    
    # Initialize network analyzer
    if 'network_analyzer' not in st.session_state:
        st.session_state.network_analyzer = NetworkTransferAnalyzer()
    
    analyzer = st.session_state.network_analyzer
    
    # Network-specific parameters
    st.markdown("### 🔧 Network Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📶 Connectivity")
        available_bandwidth = st.selectbox(
            "Available Bandwidth",
            [100, 500, 1000, 10000],
            index=2,
            format_func=lambda x: f"{x} Mbps"
        )
        
        has_direct_connect = st.checkbox("Direct Connect Available", value=False)
        has_vpn_capability = st.checkbox("VPN Capability", value=True)
    
    with col2:
        st.markdown("#### 🔒 Security Requirements")
        security_level = st.selectbox(
            "Security Requirements",
            ["standard", "high", "very_high"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        compliance_requirements = st.multiselect(
            "Compliance Requirements",
            ["HIPAA", "PCI-DSS", "SOX", "GDPR", "None"],
            default=["None"]
        )
    
    with col3:
        st.markdown("#### 💰 Budget Constraints")
        budget_level = st.selectbox(
            "Budget Level",
            ["low", "medium", "high"],
            index=1,
            format_func=lambda x: x.title()
        )
        
        max_setup_cost = st.number_input(
            "Maximum Setup Cost ($)",
            min_value=1000,
            max_value=100000,
            value=25000
        )
    
    # Update migration params with network-specific settings
    network_params = st.session_state.migration_params.copy()
    network_params.update({
        'bandwidth_mbps': available_bandwidth,
        'has_direct_connect': has_direct_connect,
        'has_vpn_capability': has_vpn_capability,
        'security_requirements': security_level,
        'compliance_requirements': compliance_requirements,
        'budget_constraints': budget_level,
        'max_setup_cost': max_setup_cost
    })
    
    # Run network analysis
    if st.button("🚀 Analyze Network Transfer Options", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing network transfer patterns..."):
            
            transfer_analysis = analyzer.calculate_transfer_analysis(network_params)
            st.session_state.transfer_analysis = transfer_analysis
            
            st.success("✅ Network analysis complete!")
    
    # Display results if available
    if hasattr(st.session_state, 'transfer_analysis') and st.session_state.transfer_analysis is not None:
        show_network_analysis_results()
    else:
        st.info("ℹ️ Run the network analysis to see results and recommendations.")
    

def show_network_analysis_results():
    """Display network analysis results"""
    
    # Fix: Check if transfer_analysis exists and is not None
    if not hasattr(st.session_state, 'transfer_analysis') or st.session_state.transfer_analysis is None:
        st.error("No network analysis results available. Please run the network analysis first.")
        return
    
    transfer_analysis = st.session_state.transfer_analysis
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Recommendations",
        "📊 Pattern Comparison", 
        "💰 Cost Analysis",
        "⏱️ Timeline Analysis",
        "🏗️ Architecture"
    ])
    
    with tab1:
        show_network_recommendations(transfer_analysis)
    
    with tab2:
        show_pattern_comparison(transfer_analysis)
    
    with tab3:
        show_network_cost_analysis(transfer_analysis)
    
    with tab4:
        show_network_timeline_analysis(transfer_analysis)
    
    with tab5:
        show_network_architecture(transfer_analysis)

def show_network_recommendations(transfer_analysis: Dict):
    """Show network recommendations"""
    
    st.markdown("### 🎯 AI-Powered Network Recommendations")
    
    # Fix: Add None check before accessing
    if transfer_analysis is None:
        st.error("No transfer analysis data available")
        return
    
    recommendations = transfer_analysis.get('recommendations', {})
    
    if not recommendations:
        st.error("No recommendations available")
        return
    
    # Primary recommendation
    primary = recommendations.get('primary_recommendation', {})
    
    if primary:
        st.markdown(f"""
        <div class="ai-insight-card">
            <h3>🏆 Primary Recommendation: {primary['pattern_name']}</h3>
            <p><strong>Score:</strong> {primary['score']:.1f}/100</p>
            <p><strong>Description:</strong> {primary['description']}</p>
            <p><strong>Reasoning:</strong> {primary['reasoning']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Implementation details
        st.markdown("#### 📋 Implementation Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Implementation Steps:**")
            for i, step in enumerate(primary.get('implementation_steps', []), 1):
                st.markdown(f"{i}. {step}")
        
        with col2:
            st.markdown(f"**Estimated Timeline:** {primary.get('estimated_timeline', 'TBD')}")
            st.markdown("**Key Benefits:**")
            for benefit in primary.get('key_considerations', []):
                st.markdown(f"• {benefit}")
    
    # Alternative options
    alternatives = recommendations.get('alternative_options', [])
    
    if alternatives:
        st.markdown("#### 🔄 Alternative Options")
        
        for i, alt in enumerate(alternatives, 1):
            with st.expander(f"Alternative {i}: {alt['pattern_name']} (Score: {alt['score']:.1f})"):
                st.markdown(f"**Best for:** {alt['best_for']}")
                st.markdown(f"**Trade-off:** {alt['trade_off']}")
    
    # Cost optimization tips
    cost_tips = recommendations.get('cost_optimization', [])
    
    if cost_tips:
        st.markdown("#### 💡 Cost Optimization Tips")
        for tip in cost_tips:
            st.markdown(f"• {tip}")
    
    # Risk considerations
    risks = recommendations.get('risk_considerations', [])
    
    if risks:
        st.markdown("#### ⚠️ Risk Considerations")
        for risk in risks:
            st.markdown(f"• {risk}")

def show_pattern_comparison(transfer_analysis: Dict):
    """Show pattern comparison visualizations"""
    
    st.markdown("### 📊 Network Pattern Comparison")
    
    # Comparison chart
    comparison_fig = create_network_comparison_chart(transfer_analysis)
    st.plotly_chart(comparison_fig, use_container_width=True, key="network_comparison_chart")
    
    # Cost vs Duration optimization
    optimization_fig = create_cost_duration_optimization_chart(transfer_analysis)
    st.plotly_chart(optimization_fig, use_container_width=True, key="cost_duration_optimization_chart")
    
    # Detailed comparison table
    st.markdown("#### 📋 Detailed Metrics Comparison")
    
    analyzer = NetworkTransferAnalyzer()
    comparison_data = []
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        pattern_info = analyzer.transfer_patterns[pattern_id]
        
        comparison_data.append({
            'Pattern': pattern_info['name'],
            'Total Cost': f"${metrics['total_cost']:,.0f}",
            'Transfer Duration': f"{metrics['transfer_time_days']:.1f} days",
            'Setup Cost': f"${metrics['setup_cost']:,.0f}",
            'Data Transfer Cost': f"${metrics['data_transfer_cost']:,.0f}",
            'Reliability Score': f"{metrics['reliability_score']}/100",
            'Security Score': f"{metrics['security_score']}/100",
            'Complexity': pattern_info['complexity']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

def show_network_timeline_analysis(transfer_analysis: Dict):
    """Show network timeline analysis"""
    
    st.markdown("### ⏱️ Timeline Analysis")
    
    recommendations = transfer_analysis.get('recommendations', {})
    timeline_impact = recommendations.get('timeline_impact', {})
    
    if timeline_impact:
        col1, col2 = st.columns(2)
        
        with col1:
            fastest = timeline_impact.get('fastest_option', {})
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #38a169;">
                <div class="metric-value" style="color: #38a169;">
                    ⚡ Fastest Option
                </div>
                <div class="metric-label">{fastest.get('pattern', 'N/A')}</div>
                <div style="margin-top: 10px;">
                    Duration: {fastest.get('duration_days', 0):.1f} days<br>
                    Timeline Usage: {fastest.get('timeline_utilization', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            slowest = timeline_impact.get('slowest_option', {})
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #e53e3e;">
                <div class="metric-value" style="color: #e53e3e;">
                    🐌 Slowest Option
                </div>
                <div class="metric-label">{slowest.get('pattern', 'N/A')}</div>
                <div style="margin-top: 10px;">
                    Duration: {slowest.get('duration_days', 0):.1f} days<br>
                    Timeline Usage: {slowest.get('timeline_utilization', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.info(f"💡 {timeline_impact.get('recommendation', '')}")
    
    # Timeline comparison chart
    patterns = []
    durations = []
    colors = []
    
    migration_timeline_days = st.session_state.migration_params.get('migration_timeline_weeks', 12) * 7
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        analyzer = NetworkTransferAnalyzer()
        pattern_info = analyzer.transfer_patterns[pattern_id]
        
        patterns.append(pattern_info['name'])
        durations.append(metrics['transfer_time_days'])
        
        # Color based on timeline fit
        if metrics['transfer_time_days'] <= migration_timeline_days * 0.3:
            colors.append('green')
        elif metrics['transfer_time_days'] <= migration_timeline_days * 0.5:
            colors.append('orange')
        else:
            colors.append('red')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=patterns,
        y=durations,
        marker_color=colors,
        text=[f"{d:.1f} days" for d in durations],
        textposition='auto'
    ))
    
    # Add timeline constraint line
    fig.add_hline(
        y=migration_timeline_days * 0.3,
        line_dash="dash",
        line_color="green",
        annotation_text="Recommended (30% of timeline)"
    )
    
    fig.update_layout(
        title='Transfer Duration vs Timeline Constraints',
        xaxis_title='Network Pattern',
        yaxis_title='Duration (days)',
        height=500,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True, key="network_timeline_duration_chart")

def show_network_architecture(transfer_analysis: Dict):
    """Show network architecture diagrams"""
    
    st.markdown("### 🏗️ Network Architecture")
    
    # Pattern selector
    analyzer = NetworkTransferAnalyzer()
    pattern_options = list(analyzer.transfer_patterns.keys())
    pattern_names = [analyzer.transfer_patterns[p]['name'] for p in pattern_options]
    
    selected_pattern_name = st.selectbox(
        "Select Pattern to Visualize",
        pattern_names
    )
    
    selected_pattern_id = pattern_options[pattern_names.index(selected_pattern_name)]
    
    # Architecture diagram
    arch_fig = create_network_architecture_diagram(selected_pattern_id)
    st.plotly_chart(arch_fig, use_container_width=True, key="network_architecture_chart")
    
    # Pattern details
    pattern_info = analyzer.transfer_patterns[selected_pattern_id]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Components")
        for component in pattern_info['components']:
            st.markdown(f"• {component}")
    
    with col2:
        st.markdown("#### 📋 Use Cases")
        for use_case in pattern_info['use_cases']:
            st.markdown(f"• {use_case}")
    
    # Pros and Cons
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Advantages")
        for pro in pattern_info['pros']:
            st.markdown(f"• {pro}")
    
    with col2:
        st.markdown("#### ⚠️ Considerations")
        for con in pattern_info['cons']:
            st.markdown(f"• {con}")
    
    # Technical specifications
    if selected_pattern_id in transfer_analysis:
        metrics = transfer_analysis[selected_pattern_id]
        
        st.markdown("#### 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Bandwidth Utilization", f"{metrics['bandwidth_utilization']}%")
        
        with col2:
            st.metric("Reliability Score", f"{metrics['reliability_score']}/100")
        
        with col3:
            st.metric("Security Score", f"{metrics['security_score']}/100")
        
        with col4:
            st.metric("Complexity Level", pattern_info['complexity'])

def show_network_cost_analysis(transfer_analysis: Dict):
    """Show detailed network cost analysis"""
    
    st.markdown("### 💰 Network Cost Analysis")
    
    # Cost breakdown for each pattern
    analyzer = NetworkTransferAnalyzer()
    chart_counter = 0
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        pattern_info = analyzer.transfer_patterns[pattern_id]
        
        with st.expander(f"💵 {pattern_info['name']} - Total Cost: ${metrics['total_cost']:,.0f}"):
             
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Setup Cost", f"${metrics['setup_cost']:,.0f}")
                st.metric("Data Transfer Cost", f"${metrics['data_transfer_cost']:,.0f}")
            
            with col2:
                st.metric("Infrastructure Cost", f"${metrics['infrastructure_cost']:,.0f}")
                st.metric("Total Cost", f"${metrics['total_cost']:,.0f}")
            
            with col3:
                st.metric("Cost per GB", f"${metrics['total_cost']/st.session_state.migration_params.get('data_size_gb', 1000):.2f}")
                
                # ROI calculation
                migration_params = st.session_state.migration_params
                data_size = migration_params.get('data_size_gb', 1000)
                
                # Estimate ongoing monthly savings (simplified)
                monthly_savings = data_size * 0.05  # Assume $0.05/GB monthly savings
                roi_months = metrics['total_cost'] / monthly_savings if monthly_savings > 0 else float('inf')
                
                if roi_months < 100:
                    st.metric("ROI Timeline", f"{roi_months:.1f} months")
                else:
                    st.metric("ROI Timeline", "Not applicable")

def show_network_timeline_analysis(transfer_analysis: Dict):
    """Show network timeline analysis"""
    
    st.markdown("### ⏱️ Timeline Analysis")
    
    recommendations = transfer_analysis.get('recommendations', {})
    timeline_impact = recommendations.get('timeline_impact', {})
    
    if timeline_impact:
        col1, col2 = st.columns(2)
        
        with col1:
            fastest = timeline_impact.get('fastest_option', {})
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #38a169;">
                <div class="metric-value" style="color: #38a169;">
                    ⚡ Fastest Option
                </div>
                <div class="metric-label">{fastest.get('pattern', 'N/A')}</div>
                <div style="margin-top: 10px;">
                    Duration: {fastest.get('duration_days', 0):.1f} days<br>
                    Timeline Usage: {fastest.get('timeline_utilization', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            slowest = timeline_impact.get('slowest_option', {})
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #e53e3e;">
                <div class="metric-value" style="color: #e53e3e;">
                    🐌 Slowest Option
                </div>
                <div class="metric-label">{slowest.get('pattern', 'N/A')}</div>
                <div style="margin-top: 10px;">
                    Duration: {slowest.get('duration_days', 0):.1f} days<br>
                    Timeline Usage: {slowest.get('timeline_utilization', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.info(f"💡 {timeline_impact.get('recommendation', '')}")
    
    # Timeline comparison chart
    patterns = []
    durations = []
    colors = []
    
    migration_timeline_days = st.session_state.migration_params.get('migration_timeline_weeks', 12) * 7
    
    for pattern_id, metrics in transfer_analysis.items():
        if pattern_id == 'recommendations':
            continue
            
        analyzer = NetworkTransferAnalyzer()
        pattern_info = analyzer.transfer_patterns[pattern_id]
        
        patterns.append(pattern_info['name'])
        durations.append(metrics['transfer_time_days'])
        
        # Color based on timeline fit
        if metrics['transfer_time_days'] <= migration_timeline_days * 0.3:
            colors.append('green')
        elif metrics['transfer_time_days'] <= migration_timeline_days * 0.5:
            colors.append('orange')
        else:
            colors.append('red')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=patterns,
        y=durations,
        marker_color=colors,
        text=[f"{d:.1f} days" for d in durations],
        textposition='auto'
    ))
    
    # Add timeline constraint line
    fig.add_hline(
        y=migration_timeline_days * 0.3,
        line_dash="dash",
        line_color="green",
        annotation_text="Recommended (30% of timeline)"
    )
    
    fig.update_layout(
        title='Transfer Duration vs Timeline Constraints',
        xaxis_title='Network Pattern',
        yaxis_title='Duration (days)',
        height=500,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_network_architecture(transfer_analysis: Dict):
    """Show network architecture diagrams"""
    
    st.markdown("### 🏗️ Network Architecture")
    
    # Pattern selector
    analyzer = NetworkTransferAnalyzer()
    pattern_options = list(analyzer.transfer_patterns.keys())
    pattern_names = [analyzer.transfer_patterns[p]['name'] for p in pattern_options]
    
    selected_pattern_name = st.selectbox(
        "Select Pattern to Visualize",
        pattern_names
    )
    
    selected_pattern_id = pattern_options[pattern_names.index(selected_pattern_name)]
    
    # Architecture diagram
    arch_fig = create_network_architecture_diagram(selected_pattern_id)
    st.plotly_chart(arch_fig, use_container_width=True)
    
    # Pattern details
    pattern_info = analyzer.transfer_patterns[selected_pattern_id]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Components")
        for component in pattern_info['components']:
            st.markdown(f"• {component}")
    
    with col2:
        st.markdown("#### 📋 Use Cases")
        for use_case in pattern_info['use_cases']:
            st.markdown(f"• {use_case}")
    
    # Pros and Cons
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Advantages")
        for pro in pattern_info['pros']:
            st.markdown(f"• {pro}")
    
    with col2:
        st.markdown("#### ⚠️ Considerations")
        for con in pattern_info['cons']:
            st.markdown(f"• {con}")
    
    # Technical specifications
    if selected_pattern_id in transfer_analysis:
        metrics = transfer_analysis[selected_pattern_id]
        
        st.markdown("#### 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Bandwidth Utilization", f"{metrics['bandwidth_utilization']}%")
        
        with col2:
            st.metric("Reliability Score", f"{metrics['reliability_score']}/100")
        
        with col3:
            st.metric("Security Score", f"{metrics['security_score']}/100")
        
        with col4:
            st.metric("Complexity Level", pattern_info['complexity'])

# Add this to the main navigation in the original app
def add_network_module_to_main_app():
    """Instructions for adding network module to main app"""
    
    # Add this to the sidebar radio options:
    # "🌐 Network Analysis"
    
    # Add this to the main content section:
    # elif page == "🌐 Network Analysis":
    #     show_network_transfer_analysis()
    
    
    pass

  # # ===========================
# ROBUST RISK ASSESSMENT FIX
# ===========================

def ensure_risk_assessment_exists():
    """Ensure risk assessment exists in session state with valid data"""
    
    if not hasattr(st.session_state, 'risk_assessment') or st.session_state.risk_assessment is None:
        st.session_state.risk_assessment = create_default_risk_assessment()
        return False  # Indicates we created a default
    return True  # Indicates existing assessment was found

def create_default_risk_assessment():
    """Create a default risk assessment when none exists"""
    
    # Try to use actual migration params if available
    migration_params = getattr(st.session_state, 'migration_params', {})
    recommendations = getattr(st.session_state, 'recommendations', {}) or getattr(st.session_state, 'enhanced_recommendations', {})
    
    # Calculate risks based on available data
    try:
        if migration_params and recommendations:
            return calculate_migration_risks_safe(migration_params, recommendations)
        else:
            return get_fallback_risk_assessment()
    except:
        return get_fallback_risk_assessment()

def calculate_migration_risks_safe(migration_params: Dict, recommendations: Dict) -> Dict:
    """Safe version of risk calculation that never fails"""
    
    try:
        # Get basic parameters with safe defaults
        source_engine = migration_params.get('source_engine', 'unknown')
        target_engine = migration_params.get('target_engine', 'unknown')
        data_size_gb = int(migration_params.get('data_size_gb', 1000))
        timeline_weeks = int(migration_params.get('migration_timeline_weeks', 12))
        team_size = int(migration_params.get('team_size', 5))
        migration_budget = int(migration_params.get('migration_budget', 500000))
        
        # Calculate technical risks
        engine_risk = calculate_engine_compatibility_risk(source_engine, target_engine)
        data_risk = calculate_data_complexity_risk(data_size_gb)
        app_risk = calculate_application_risk(migration_params)
        perf_risk = calculate_performance_risk(recommendations)
        
        technical_risks = {
            'engine_compatibility': engine_risk,
            'data_migration_complexity': data_risk,
            'application_integration': app_risk,
            'performance_risk': perf_risk
        }
        
        # Calculate business risks
        timeline_risk = calculate_timeline_risk(timeline_weeks, data_size_gb, team_size)
        cost_risk = calculate_cost_risk(migration_budget, data_size_gb, timeline_weeks)
        continuity_risk = calculate_continuity_risk(recommendations)
        resource_risk = calculate_resource_risk(team_size, migration_params.get('team_expertise', 'medium'))
        
        business_risks = {
            'timeline_risk': timeline_risk,
            'cost_overrun_risk': cost_risk,
            'business_continuity': continuity_risk,
            'resource_availability': resource_risk
        }
        
        # Calculate overall score
        tech_avg = sum(technical_risks.values()) / len(technical_risks)
        business_avg = sum(business_risks.values()) / len(business_risks)
        overall_score = (tech_avg + business_avg) / 2
        
        # Generate risk level
        risk_level = get_risk_level_safe(overall_score)
        
        # Generate mitigation strategies
        mitigation_strategies = generate_mitigation_strategies_safe(technical_risks, business_risks)
        
        return {
            'overall_score': overall_score,
            'risk_level': risk_level,
            'technical_risks': technical_risks,
            'business_risks': business_risks,
            'mitigation_strategies': mitigation_strategies
        }
        
    except Exception as e:
        print(f"Error in safe risk calculation: {e}")
        return get_fallback_risk_assessment()

def calculate_engine_compatibility_risk(source: str, target: str) -> float:
    """Calculate engine compatibility risk safely"""
    compatibility_scores = {
        ('oracle-ee', 'postgres'): 75,
        ('oracle-ee', 'aurora-postgresql'): 65,
        ('oracle-se', 'postgres'): 70,
        ('postgres', 'aurora-postgresql'): 20,
        ('postgres', 'postgres'): 10,
        ('mysql', 'aurora-mysql'): 15,
        ('mysql', 'mysql'): 10,
        ('sql-server', 'postgres'): 80
    }
    
    # Same engine = low risk
    if source == target:
        return 15
    
    return compatibility_scores.get((source, target), 50)

def calculate_data_complexity_risk(data_size_gb: int) -> float:
    """Calculate data complexity risk safely"""
    if data_size_gb < 100:
        return 20
    elif data_size_gb < 1000:
        return 40
    elif data_size_gb < 10000:
        return 60
    else:
        return 80

def calculate_application_risk(migration_params: Dict) -> float:
    """Calculate application integration risk safely"""
    num_apps = migration_params.get('num_applications', 1)
    num_procedures = migration_params.get('num_stored_procedures', 0)
    
    base_risk = min(80, 20 + (num_apps * 10))
    procedure_risk = min(25, num_procedures / 20)
    
    return min(90, base_risk + procedure_risk)

def calculate_performance_risk(recommendations: Dict) -> float:
    """Calculate performance risk safely"""
    if not recommendations:
        return 50
    
    # Count production environments
    prod_envs = 0
    adequate_sizing = 0
    
    for env_name, rec in recommendations.items():
        env_type = rec.get('environment_type', '').lower()
        if 'prod' in env_type or env_type == 'production':
            prod_envs += 1
            
            # Check instance sizing
            if 'writer' in rec:  # Enhanced recommendations
                instance_class = rec['writer'].get('instance_class', '')
            else:  # Regular recommendations
                instance_class = rec.get('instance_class', '')
            
            if 'xlarge' in instance_class or 'large' in instance_class:
                adequate_sizing += 1
    
    if prod_envs == 0:
        return 30  # No production = lower risk
    
    adequacy_ratio = adequate_sizing / prod_envs
    return 70 - (adequacy_ratio * 40)  # Better sizing = lower risk

def calculate_timeline_risk(timeline_weeks: int, data_size_gb: int, team_size: int) -> float:
    """Calculate timeline risk safely"""
    base_risk = 30
    
    # Timeline pressure
    if timeline_weeks < 8:
        base_risk += 30
    elif timeline_weeks < 12:
        base_risk += 15
    
    # Data size impact
    if data_size_gb > 10000:
        base_risk += 20
    elif data_size_gb > 5000:
        base_risk += 10
    
    # Team size impact
    if team_size < 3:
        base_risk += 20
    elif team_size < 5:
        base_risk += 10
    
    return min(90, base_risk)

def calculate_cost_risk(budget: int, data_size_gb: int, timeline_weeks: int) -> float:
    """Calculate cost risk safely"""
    if budget <= 0:
        return 60  # Unknown budget
    
    # Rough cost estimation
    estimated_cost = (data_size_gb * 50) + (timeline_weeks * 8000)
    
    ratio = estimated_cost / budget
    if ratio < 0.5:
        return 20
    elif ratio < 0.8:
        return 40
    elif ratio < 1.0:
        return 60
    else:
        return 85

def calculate_continuity_risk(recommendations: Dict) -> float:
    """Calculate business continuity risk safely"""
    if not recommendations:
        return 50
    
    env_count = len(recommendations)
    
    if env_count == 1:
        return 40
    elif env_count < 4:
        return 50
    else:
        return 60  # More environments = more coordination complexity

def calculate_resource_risk(team_size: int, expertise: str) -> float:
    """Calculate resource availability risk safely"""
    base_risk = 40
    
    if team_size < 3:
        base_risk += 25
    elif team_size < 5:
        base_risk += 10
    elif team_size > 10:
        base_risk += 5  # Too large can also be a problem
    
    if expertise == 'high':
        base_risk -= 15
    elif expertise == 'low':
        base_risk += 20
    
    return max(15, min(85, base_risk))

def get_risk_level_safe(score: float) -> Dict:
    """Get risk level safely"""
    if score < 30:
        return {'level': 'Low', 'color': '#38a169', 'action': 'Standard monitoring sufficient'}
    elif score < 50:
        return {'level': 'Medium', 'color': '#d69e2e', 'action': 'Active monitoring recommended'}
    elif score < 70:
        return {'level': 'High', 'color': '#e53e3e', 'action': 'Immediate mitigation required'}
    else:
        return {'level': 'Critical', 'color': '#9f1239', 'action': 'Project review required'}

def generate_mitigation_strategies_safe(technical_risks: Dict, business_risks: Dict) -> List[Dict]:
    """Generate mitigation strategies safely"""
    strategies = []
    
    # Technical strategies
    if technical_risks.get('engine_compatibility', 0) > 60:
        strategies.append({
            'risk': 'Engine Compatibility',
            'strategy': 'Use AWS Schema Conversion Tool and conduct thorough testing',
            'timeline': '2-3 weeks',
            'cost_impact': 'Medium'
        })
    
    if technical_risks.get('data_migration_complexity', 0) > 50:
        strategies.append({
            'risk': 'Data Migration Complexity',
            'strategy': 'Implement phased migration with comprehensive validation',
            'timeline': '1-2 weeks',
            'cost_impact': 'Low'
        })
    
    # Business strategies
    if business_risks.get('timeline_risk', 0) > 60:
        strategies.append({
            'risk': 'Timeline Risk',
            'strategy': 'Add parallel workstreams and increase team capacity',
            'timeline': 'Immediate',
            'cost_impact': 'High'
        })
    
    if business_risks.get('cost_overrun_risk', 0) > 60:
        strategies.append({
            'risk': 'Cost Management',
            'strategy': 'Implement strict budget controls and milestone reviews',
            'timeline': 'Ongoing',
            'cost_impact': 'Low'
        })
    
    # Default strategy if none generated
    if not strategies:
        strategies.append({
            'risk': 'General Migration Management',
            'strategy': 'Follow AWS migration best practices and maintain regular checkpoints',
            'timeline': '2-3 weeks',
            'cost_impact': 'Medium'
        })
    
    return strategies

def get_fallback_risk_assessment() -> Dict:
    """Fallback risk assessment when all else fails"""
    return {
        'overall_score': 45,
        'risk_level': {'level': 'Medium', 'color': '#d69e2e', 'action': 'Standard migration practices recommended'},
        'technical_risks': {
            'engine_compatibility': 40,
            'data_migration_complexity': 35,
            'application_integration': 45,
            'performance_risk': 30
        },
        'business_risks': {
            'timeline_risk': 50,
            'cost_overrun_risk': 40,
            'business_continuity': 45,
            'resource_availability': 35
        },
        'mitigation_strategies': [
            {
                'risk': 'Migration Planning',
                'strategy': 'Develop comprehensive migration plan with clear milestones',
                'timeline': '1-2 weeks',
                'cost_impact': 'Low'
            },
            {
                'risk': 'Testing & Validation',
                'strategy': 'Implement thorough testing at each migration phase',
                'timeline': '2-3 weeks',
                'cost_impact': 'Medium'
            }
        ]
    }

# UPDATED show_risk_assessment function
def show_risk_assessment_robust():
    """Show risk assessment dashboard - ROBUST VERSION"""
    
    st.markdown("### ⚠️ Migration Risk Assessment")
    
    # Ensure risk assessment exists
    risk_exists = ensure_risk_assessment_exists()
    
    if not risk_exists:
        st.info("📊 Risk assessment generated from available data")
    
    # Add manual refresh option
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Risk Assessment"):
            st.session_state.risk_assessment = create_default_risk_assessment()
            st.success("✅ Risk assessment refreshed!")
            st.experimental_rerun()
    
    # Debug information (optional)
    if st.checkbox("🔍 Show Risk Debug Info"):
        st.write("Risk assessment status:", "✅ Available" if st.session_state.risk_assessment else "❌ Missing")
        if st.session_state.risk_assessment:
            st.write("Overall score:", st.session_state.risk_assessment.get('overall_score', 'N/A'))
            st.write("Technical risks count:", len(st.session_state.risk_assessment.get('technical_risks', {})))
            st.write("Business risks count:", len(st.session_state.risk_assessment.get('business_risks', {})))
    
    # Get risk assessment data
    risk_assessment = st.session_state.risk_assessment
    
    # Overall risk level display
    risk_level = risk_assessment.get('risk_level', {'level': 'Unknown', 'color': '#666666', 'action': 'Assessment needed'})
    overall_score = risk_assessment.get('overall_score', 50)
    
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {risk_level['color']}; background: linear-gradient(90deg, {risk_level['color']}22, white);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 2rem; font-weight: bold; color: {risk_level['color']};">
                    {risk_level['level']} Risk
                </div>
                <div style="font-size: 1.1rem; color: #666; margin: 5px 0;">
                    Overall Score: {overall_score:.1f}/100
                </div>
                <div style="font-weight: 500; color: #333;">
                    📋 {risk_level['action']}
                </div>
            </div>
            <div style="font-size: 3rem;">
                {'🟢' if overall_score < 30 else '🟡' if overall_score < 50 else '🟠' if overall_score < 70 else '🔴'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Technical Risks")
        tech_risks = risk_assessment.get('technical_risks', {})
        
        for risk_name, score in tech_risks.items():
            risk_level_detail = 'High' if score > 60 else 'Medium' if score > 30 else 'Low'
            color = '#e53e3e' if score > 60 else '#d69e2e' if score > 30 else '#38a169'
            
            # Progress bar style display
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span><strong>{risk_name.replace('_', ' ').title()}</strong></span>
                    <span style="color: {color}; font-weight: bold;">{score:.0f}/100</span>
                </div>
                <div style="background: #f0f0f0; border-radius: 10px; height: 8px; margin: 5px 0;">
                    <div style="background: {color}; width: {score}%; height: 8px; border-radius: 10px;"></div>
                </div>
                <div style="color: {color}; font-size: 0.9rem; font-weight: 500;">{risk_level_detail} Risk</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 💼 Business Risks")
        business_risks = risk_assessment.get('business_risks', {})
        
        for risk_name, score in business_risks.items():
            risk_level_detail = 'High' if score > 60 else 'Medium' if score > 30 else 'Low'
            color = '#e53e3e' if score > 60 else '#d69e2e' if score > 30 else '#38a169'
            
            # Progress bar style display
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span><strong>{risk_name.replace('_', ' ').title()}</strong></span>
                    <span style="color: {color}; font-weight: bold;">{score:.0f}/100</span>
                </div>
                <div style="background: #f0f0f0; border-radius: 10px; height: 8px; margin: 5px 0;">
                    <div style="background: {color}; width: {score}%; height: 8px; border-radius: 10px;"></div>
                </div>
                <div style="color: {color}; font-size: 0.9rem; font-weight: 500;">{risk_level_detail} Risk</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Risk mitigation strategies
    st.markdown("---")
    st.markdown("#### 🛡️ Risk Mitigation Strategies")
    
    mitigation_strategies = risk_assessment.get('mitigation_strategies', [])
    
    if mitigation_strategies:
        for i, strategy in enumerate(mitigation_strategies, 1):
            with st.expander(f"🎯 Strategy {i}: {strategy.get('risk', 'Migration Risk')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Strategy:** {strategy.get('strategy', 'Not specified')}")
                    
                with col2:
                    st.markdown(f"**Timeline:** {strategy.get('timeline', 'TBD')}")
                    
                    cost_impact = strategy.get('cost_impact', 'Medium')
                    cost_color = '#e53e3e' if cost_impact == 'High' else '#d69e2e' if cost_impact == 'Medium' else '#38a169'
                    st.markdown(f"**Cost Impact:** <span style='color: {cost_color}; font-weight: bold;'>{cost_impact}</span>", unsafe_allow_html=True)
    else:
        st.info("✅ Current risk levels are manageable with standard migration best practices.")

# UPDATED run_migration_analysis function
def run_migration_analysis_robust():
    """Run comprehensive migration analysis - ROBUST VERSION"""
    
    try:
        # Initialize analyzer
        anthropic_api_key = st.session_state.migration_params.get('anthropic_api_key')
        analyzer = MigrationAnalyzer(anthropic_api_key)
        
        # Step 1: Calculate recommendations
        st.write("📊 Calculating instance recommendations...")
        recommendations = analyzer.calculate_instance_recommendations(st.session_state.environment_specs)
        st.session_state.recommendations = recommendations
        
        # Step 2: Calculate costs
        st.write("💰 Analyzing costs...")
        cost_analysis = analyzer.calculate_migration_costs(recommendations, st.session_state.migration_params)
        
        # Update migration params with estimated cost
        st.session_state.migration_params['estimated_migration_cost'] = cost_analysis['migration_costs']['total']
        st.session_state.analysis_results = cost_analysis
        
        # Step 3: Risk assessment - ROBUST VERSION
        st.write("⚠️ Assessing risks...")
        
        # Force create risk assessment
        risk_assessment = create_default_risk_assessment()
        st.session_state.risk_assessment = risk_assessment
        st.write("✅ Risk assessment completed successfully")
        
      # Step 4: AI insights
        if anthropic_api_key:
            st.write("🤖 Generating AI insights...")
            try:
                ai_insights = analyzer.generate_ai_insights_sync(cost_analysis, st.session_state.migration_params)
                
                if ai_insights.get('success'):
                    st.success("✅ Real Claude AI analysis complete!")
                    st.info(f"Model: {ai_insights.get('model', 'Unknown')}")
                else:
                    st.warning(f"⚠️ Claude AI failed: {ai_insights.get('error')}")
                
                st.session_state.ai_insights = ai_insights
                
            except Exception as e:
                st.warning(f"AI insights generation failed: {str(e)}")
                st.session_state.ai_insights = {'error': str(e)}
        else:
            st.info("ℹ️ Provide Anthropic API key for Claude AI insights")
        
        st.success("✅ Analysis complete!")
        
        # Show quick summary
        st.markdown("#### 🎯 Analysis Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monthly Cost", f"${cost_analysis['monthly_aws_cost']:,.0f}")
        
        with col2:
            st.metric("Migration Cost", f"${cost_analysis['migration_costs']['total']:,.0f}")
        
        with col3:
            risk_level = st.session_state.risk_assessment['risk_level']['level']
            st.metric("Risk Level", risk_level)
        
        # Provide navigation hint
        st.info("📈 View detailed results in the 'Results Dashboard' section")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        
        # Even if analysis fails, ensure we have a risk assessment
        if not hasattr(st.session_state, 'risk_assessment') or st.session_state.risk_assessment is None:
            st.session_state.risk_assessment = get_fallback_risk_assessment()
            st.info("🛡️ Fallback risk assessment created")
            
# ===========================
# VISUALIZATION FUNCTIONS
# ===========================

def create_cost_waterfall_chart(cost_analysis: Dict) -> go.Figure:
    """Create cost transformation waterfall chart"""
    
    current_costs = cost_analysis.get('current_total_cost', 0)
    aws_costs = cost_analysis['annual_aws_cost']
    migration_costs = cost_analysis['migration_costs']['total']
    
    # Waterfall components
    categories = ['Current Infrastructure', 'Migration Investment', 'AWS Annual Cost', 'Net Position']
    values = [current_costs, -migration_costs, -aws_costs, current_costs - migration_costs - aws_costs]
    
    fig = go.Figure()
    
    # Create waterfall effect
    cumulative = [current_costs]
    for i in range(1, len(values)-1):
        cumulative.append(cumulative[-1] + values[i])
    cumulative.append(values[-1])
    
    colors = ['blue', 'red', 'orange', 'green' if values[-1] > 0 else 'red']
    
    for i, (category, value, color) in enumerate(zip(categories, values, colors)):
        fig.add_trace(go.Bar(
            x=[category],
            y=[abs(value)],
            name=category,
            marker_color=color,
            text=f'${abs(value):,.0f}',
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Cost Transformation Analysis',
        xaxis_title='Cost Components',
        yaxis_title='Annual Cost ($)',
        showlegend=False,
        height=500
    )
    
    return fig

def create_risk_heatmap(risk_assessment: Dict) -> go.Figure:
    """Create risk assessment heatmap"""
    
    # Prepare risk data
    tech_risks = risk_assessment['technical_risks']
    business_risks = risk_assessment['business_risks']
    
    risk_categories = list(tech_risks.keys()) + list(business_risks.keys())
    risk_scores = list(tech_risks.values()) + list(business_risks.values())
    risk_types = ['Technical'] * len(tech_risks) + ['Business'] * len(business_risks)
    
    # Create heatmap data
    heatmap_data = []
    for i, (category, score, risk_type) in enumerate(zip(risk_categories, risk_scores, risk_types)):
        heatmap_data.append([score])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=['Risk Score'],
        y=[cat.replace('_', ' ').title() for cat in risk_categories],
        colorscale=[
            [0, '#38a169'],      # Green
            [0.3, '#d69e2e'],    # Yellow
            [0.6, '#e53e3e'],    # Red
            [1, '#9f1239']       # Dark red
        ],
        text=[[f'{score:.0f}' for score in row] for row in heatmap_data],
        texttemplate='%{text}',
        colorbar=dict(title="Risk Level")
    ))
    
    fig.update_layout(
        title='Migration Risk Assessment',
        height=600,
        yaxis=dict(autorange="reversed")
    )
    
    return fig

        
# Alternative simplified version if the above still has issues
def show_results_dashboard_simple():
    """Simplified results dashboard to avoid indentation issues"""
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ No analysis results available. Please run the migration analysis first.")
        return
    
    st.markdown("## 📊 Migration Analysis Results")
    
    # Check for enhanced results
    has_enhanced_results = (hasattr(st.session_state, 'enhanced_analysis_results') and 
                           st.session_state.enhanced_analysis_results is not None)
    
    # Create tabs for different views
    tabs = st.tabs([
        "💰 Cost Summary",
        "📈 Growth Projections", 
        "💎 Enhanced Analysis",
        "⚠️ Risk Assessment", 
        "🏢 Environment Analysis",
        "📊 Visualizations",
        "🤖 AI Insights",
        "📅 Timeline"
    ])
    
    # Tab content
    with tabs[0]:  # Cost Summary
        show_basic_cost_summary()
    
    with tabs[1]:  # Growth Projections
        show_growth_analysis_dashboard()
    
    with tabs[2]:  # Enhanced Analysis
        if has_enhanced_results:
            show_enhanced_cost_analysis()
        else:
            st.info("💡 Enhanced cost analysis not available. Use the enhanced environment setup to access detailed cost breakdowns.")
            show_basic_cost_summary()
    
    with tabs[3]:  # Risk Assessment
        show_risk_assessment_tab()
    
    with tabs[4]:  # Environment Analysis
        show_environment_analysis_tab()
    
    with tabs[5]:  # Visualizations
        show_visualizations_tab()
    
    with tabs[6]:  # AI Insights
        show_ai_insights_tab()
    
    with tabs[7]:  # Timeline
        show_timeline_analysis_tab()
        
        
def show_vrops_dashboard():
    """Show vROps data dashboard"""
    
    st.markdown("### 📊 vROps Data Dashboard")
    
    servers = st.session_state.vrops_servers
    db_servers = st.session_state.vrops_database_servers
    analysis = st.session_state.vrops_analysis
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Servers", len(servers))
    
    with col2:
        st.metric("Database Servers", len(db_servers))
    
    with col3:
        if analysis:
            avg_health = sum([a['health_scores']['overall'] for a in analysis.values()]) / len(analysis)
            st.metric("Avg Health Score", f"{avg_health:.1f}")
    
    with col4:
        ready_count = sum([1 for a in analysis.values() if a['migration_readiness'] == 'Ready'])
        st.metric("Migration Ready", f"{ready_count}/{len(analysis)}")
    
    # Server list
    st.markdown("#### 🖥️ Server Inventory")
    
    if servers:
        server_data = []
        for server in servers:
            analysis_data = analysis.get(server['id'], {})
            performance = analysis_data.get('performance_summary', {})
            aws_rec = analysis_data.get('aws_recommendations', {})
            
            server_data.append({
                'Server Name': server['name'],
                'CPU Cores': server['cpu_cores'],
                'Memory (GB)': f"{server['memory_gb']:.1f}",
                'CPU Usage (%)': f"{performance.get('cpu_usage_avg', 0):.1f}",
                'Memory Usage (%)': f"{performance.get('memory_usage_avg', 0):.1f}",
                'AWS Instance': aws_rec.get('instance_class', 'N/A'),
                'Health Score': f"{analysis_data.get('health_scores', {}).get('overall', 0):.1f}",
                'Migration Status': analysis_data.get('migration_readiness', 'Unknown')
            })
        
        df = pd.DataFrame(server_data)
        st.dataframe(df, use_container_width=True)
    
    # Actions
    st.markdown("#### 🎯 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Convert to Environment Specs", use_container_width=True):
            convert_vrops_to_environments()
    
    with col2:
        if st.button("📊 View Detailed Analysis", use_container_width=True):
            show_detailed_vrops_analysis()
    
    with col3:
        if st.button("🔌 Reconnect to vROps", use_container_width=True):
            # Clear data and reconnect
            for key in ['vrops_servers', 'vrops_database_servers', 'vrops_metrics', 'vrops_analysis']:
                if key in st.session_state:
                    st.session_state[key] = []
            st.experimental_rerun()


def show_vrops_results_summary():
    """Show vROps results in the results dashboard"""
    
    if not st.session_state.vrops_analysis:
        st.info("No vROps analysis data available.")
        return
    
    st.markdown("### 📡 vROps Performance Analysis")
    
    analysis = st.session_state.vrops_analysis
    servers = st.session_state.vrops_servers
    
    # Performance summary
    col1, col2, col3 = st.columns(3)
    
    health_scores = [a['health_scores']['overall'] for a in analysis.values()]
    ready_count = sum([1 for a in analysis.values() if a['migration_readiness'] == 'Ready'])
    
    with col1:
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0
        st.metric("Average Health Score", f"{avg_health:.1f}/100")
    
    with col2:
        st.metric("Migration Ready", f"{ready_count}/{len(analysis)}")
    
    with col3:
        db_servers = len(st.session_state.vrops_database_servers)
        st.metric("Database Servers", db_servers)
    
    # Health distribution chart
    if health_scores:
        fig = go.Figure()
        
        server_names = [next((s['name'] for s in servers if s['id'] == sid), sid) 
                       for sid in analysis.keys()]
        
        fig.add_trace(go.Bar(
            x=server_names,
            y=health_scores,
            marker_color=['green' if score > 80 else 'orange' if score > 60 else 'red' 
                         for score in health_scores],
            text=[f"{score:.1f}" for score in health_scores],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Server Health Scores',
            xaxis_title='Server',
            yaxis_title='Health Score',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance insights
    st.markdown("#### 💡 Performance Insights")
    
    high_cpu_count = sum([1 for a in analysis.values() 
                         if a.get('performance_summary', {}).get('cpu_usage_avg', 0) > 80])
    high_memory_count = sum([1 for a in analysis.values() 
                            if a.get('performance_summary', {}).get('memory_usage_avg', 0) > 85])
    
    if high_cpu_count > 0:
        st.warning(f"⚠️ {high_cpu_count} servers have high CPU usage (>80%)")
    
    if high_memory_count > 0:
        st.warning(f"⚠️ {high_memory_count} servers have high memory usage (>85%)")
    
    if high_cpu_count == 0 and high_memory_count == 0:
        st.success("✅ All servers show healthy performance metrics")
    

def show_basic_cost_summary():
    """Show basic cost summary from analysis results"""
    
    if not st.session_state.analysis_results:
        st.error("No analysis results available.")
        return
    
    results = st.session_state.analysis_results
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        monthly_cost = results.get('monthly_aws_cost', 0)
        st.metric("Monthly AWS Cost", f"${monthly_cost:,.0f}")
    
    with col2:
        annual_cost = monthly_cost * 12
        st.metric("Annual AWS Cost", f"${annual_cost:,.0f}")
    
    with col3:
        migration_cost = results.get('migration_costs', {}).get('total', 0)
        st.metric("Migration Cost", f"${migration_cost:,.0f}")
    
    with col4:
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            growth_percent = st.session_state.growth_analysis['growth_summary']['total_3_year_growth_percent']
            st.metric("3-Year Growth", f"{growth_percent:.1f}%")
        else:
            total_cost = annual_cost + migration_cost
            st.metric("Total First Year", f"${total_cost:,.0f}")
    
    # Environment costs breakdown
    st.markdown("### 💰 Cost Breakdown by Environment")
    
    env_costs = results.get('environment_costs', {})
    if env_costs:
        for env_name, costs in env_costs.items():
            with st.expander(f"🏢 {env_name.title()} Environment"):
                if isinstance(costs, dict) and 'instance_cost' in costs:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Instance Cost", f"${costs.get('instance_cost', 0):,.2f}/month")
                        st.metric("Storage Cost", f"${costs.get('storage_cost', 0):,.2f}/month")
                    
                    with col2:
                        st.metric("Reader Instances", f"${costs.get('reader_costs', 0):,.2f}/month")
                        st.metric("Backup Cost", f"${costs.get('backup_cost', 0):,.2f}/month")
                    
                    with col3:
                        total_env_cost = costs.get('total_monthly', 
                                                 sum([costs.get(k, 0) for k in ['instance_cost', 'storage_cost', 'reader_costs', 'backup_cost']]))
                        st.metric("Total Monthly", f"${total_env_cost:,.2f}")
                else:
                    if isinstance(costs, (int, float)):
                        st.metric("Monthly Cost", f"${costs:,.2f}")
                    else:
                        st.write("Cost information not available in expected format")


def show_growth_analysis_dashboard():
    """Show comprehensive growth analysis dashboard"""
    
    st.markdown("### 📈 3-Year Growth Analysis & Projections")
    
    # Check if growth analysis exists
    if not hasattr(st.session_state, 'growth_analysis') or not st.session_state.growth_analysis:
        st.warning("⚠️ Growth analysis not available. Please run the analysis first.")
        
        # Show basic growth planning instead
        st.markdown("#### 🎯 Growth Planning Preview")
        st.info("""
        **Growth analysis will show:**
        - 3-year cost projections with growth factors
        - Resource scaling requirements
        - Seasonal peak planning
        - Cost optimization opportunities
        - Scaling recommendations by year
        
        Run the migration analysis to see detailed growth projections.
        """)
        return
    
    growth_analysis = st.session_state.growth_analysis
    growth_summary = growth_analysis['growth_summary']
    
    # Key Growth Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "3-Year Growth",
            f"{growth_summary['total_3_year_growth_percent']:.1f}%",
            delta=f"CAGR: {growth_summary['compound_annual_growth_rate']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Current Annual Cost",
            f"${growth_summary['year_0_cost']:,.0f}",
            delta="Baseline"
        )
    
    with col3:
        st.metric(
            "Year 3 Projected Cost",
            f"${growth_summary['year_3_cost']:,.0f}",
            delta=f"+${growth_summary['year_3_cost'] - growth_summary['year_0_cost']:,.0f}"
        )
    
    with col4:
        st.metric(
            "Total 3-Year Investment",
            f"${growth_summary['total_3_year_investment']:,.0f}",
            delta=f"Avg: ${growth_summary['average_annual_cost']:,.0f}/year"
        )

     # Growth Projection Charts
    st.markdown("#### 📊 Growth Projections")
    
    try:
        charts = create_growth_projection_charts(growth_analysis)
        
        # Display charts with unique keys
        for i, chart in enumerate(charts):
            st.plotly_chart(chart, use_container_width=True, key=f"growth_chart_{i}")
            
    except Exception as e:
        st.error(f"Error creating growth charts: {str(e)}")
        
        # Show basic growth table instead
        st.markdown("#### 📈 Year-by-Year Projections")
        projections = growth_analysis['yearly_projections']
        
        years_data = []
        for year in range(4):
            year_data = projections[f'year_{year}']
            years_data.append({
                'Year': f'Year {year}' if year > 0 else 'Current',
                'Monthly Cost': f"${year_data['total_monthly']:,.0f}",
                'Annual Cost': f"${year_data['total_annual']:,.0f}",
                'Peak Cost': f"${year_data['peak_annual']:,.0f}"
            })
        
        st.table(years_data)
    
    # Scaling Recommendations
    st.markdown("#### 🎯 Scaling Recommendations")
    
    recommendations = growth_analysis.get('scaling_recommendations', [])
    
    if recommendations:
        for rec in recommendations:
            priority_color = {
                'High': '#e53e3e',
                'Medium': '#d69e2e',
                'Low': '#38a169'
            }.get(rec['priority'], '#666666')
            
            st.markdown(f"""
            <div style="border-left: 4px solid {priority_color}; padding: 15px; margin: 10px 0; background: {priority_color}22;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: {priority_color};">{rec['type']} (Year {rec['year']})</strong><br>
                        {rec['description']}<br>
                        <em>Action: {rec['action']}</em>
                    </div>
                    <div style="text-align: right;">
                        <strong>Priority: {rec['priority']}</strong><br>
                        <span style="color: #38a169;">Potential Savings: ${rec['estimated_savings']:,.0f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical scaling issues identified in the 3-year projection.")
        
        # Show default recommendations
        st.markdown("#### 💡 General Growth Recommendations")
        default_recommendations = [
            "📊 **Monitor growth trends** - Track actual vs projected growth quarterly",
            "💰 **Reserved Instances** - Consider 1-3 year commitments for 30-40% savings",
            "🔄 **Auto-scaling** - Implement auto-scaling for variable workloads",
            "📈 **Capacity planning** - Review and adjust capacity every 6 months",
            "🗄️ **Data lifecycle** - Implement archiving for older data to reduce storage costs"
        ]
        
        for rec in default_recommendations:
            st.markdown(rec)
    

def show_risk_assessment_tab():
    """Show risk assessment results"""
    
    if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
        show_risk_assessment_robust()
    else:
        st.warning("⚠️ Risk assessment not available. Please run the migration analysis first.")


def show_environment_analysis_tab():
    """Show environment analysis"""
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Environment analysis not available. Please run the migration analysis first.")
        return
        
    st.markdown("### 🏢 Environment Analysis")
    
    # Show environment specifications
    if hasattr(st.session_state, 'environment_specs') and st.session_state.environment_specs:
        st.markdown("#### 📋 Environment Specifications")
        
        for env_name, specs in st.session_state.environment_specs.items():
            with st.expander(f"🏢 {env_name.title()} Environment"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Specs:**")
                    st.write(f"CPU Cores: {specs.get('cpu_cores', 'N/A')}")
                    st.write(f"RAM: {specs.get('ram_gb', 'N/A')} GB")
                    st.write(f"Storage: {specs.get('storage_gb', 'N/A')} GB")
                    st.write(f"Daily Usage: {specs.get('daily_usage_hours', 'N/A')} hours")
                
                with col2:
                    st.markdown("**Workload:**")
                    st.write(f"Peak Connections: {specs.get('peak_connections', 'N/A')}")
                    st.write(f"Workload Pattern: {specs.get('workload_pattern', 'N/A')}")
                    st.write(f"Environment Type: {specs.get('environment_type', 'N/A')}")


    """Environment setup with vROps option"""
    
    st.markdown("## 📊 Environment Configuration")
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Please complete Migration Configuration first.")
        return
    
    # Configuration method selection
    config_method = st.radio(
        "Choose configuration method:",
        [
            "🔗 Import from vROps",  # <-- NEW OPTION
            "📝 Manual Configuration",
            "📁 Bulk Upload"
        ],
        horizontal=True
    )
    
    if config_method == "🔗 Import from vROps":
        # Check if vROps data exists
        if st.session_state.vrops_servers:
            st.success("✅ vROps data available!")
            if st.button("🔄 Use vROps Data for Environment Setup", type="primary"):
                convert_vrops_to_environments()
        else:
            st.info("No vROps data available. Please connect to vROps first.")
            if st.button("📡 Go to vROps Integration", use_container_width=True):
                st.session_state.page = "📡 vROps Integration"
                st.experimental_rerun()
    
    elif config_method == "📝 Manual Configuration":
        show_manual_environment_setup()
    
    else:  # Bulk Upload
        show_bulk_upload_interface()





def show_visualizations_tab():
    """Show visualization charts"""
    
    st.markdown("### 📊 Cost & Performance Visualizations")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Visualizations not available. Please run the migration analysis first.")
        return
    
    try:
        # Cost comparison chart
        results = st.session_state.analysis_results
        
        # Create simple cost visualization
        env_costs = results.get('environment_costs', {})
        
        if env_costs:
            # Environment cost comparison
            env_names = []
            monthly_costs = []
            
            for env_name, costs in env_costs.items():
                env_names.append(env_name.title())
                
                if isinstance(costs, dict):
                    cost = costs.get('total_monthly', 
                                   sum([costs.get(k, 0) for k in ['instance_cost', 'storage_cost', 'reader_costs', 'backup_cost']]))
                else:
                    cost = float(costs) if costs else 0
                
                monthly_costs.append(cost)
            
            if env_names and monthly_costs:
                fig = go.Figure(data=[
                    go.Bar(x=env_names, y=monthly_costs, marker_color='#3182ce')
                ])
                
                fig.update_layout(
                    title='Monthly Cost by Environment',
                    xaxis_title='Environment',
                    yaxis_title='Monthly Cost ($)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True, key="env_cost_chart")
        
        # Growth visualization if available
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            st.markdown("#### 📈 Growth Projections")
            try:
                charts = create_growth_projection_charts(st.session_state.growth_analysis)
                for i, chart in enumerate(charts):
                    st.plotly_chart(chart, use_container_width=True, key=f"viz_growth_chart_{i}")
            except Exception as e:
                st.error(f"Error creating growth charts: {str(e)}")
        
        # Enhanced cost chart if available
        if hasattr(st.session_state, 'enhanced_cost_chart') and st.session_state.enhanced_cost_chart:
            st.markdown("#### 💎 Enhanced Cost Analysis")
            st.plotly_chart(st.session_state.enhanced_cost_chart, use_container_width=True, key="enhanced_cost_chart")
        
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")


def show_ai_insights_tab():
    """Show AI insights if available"""
    
    if hasattr(st.session_state, 'ai_insights') and st.session_state.ai_insights:
        st.markdown("### 🤖 AI-Powered Insights")
        
        insights = st.session_state.ai_insights
        
        if 'error' in insights:
            st.warning(f"AI insights partially available: {insights.get('summary', 'Analysis complete')}")
            st.error(f"Error: {insights['error']}")
        else:
            if 'summary' in insights:
                st.markdown("#### 📝 Executive Summary")
                st.write(insights['summary'])
            
            if 'recommendations' in insights:
                st.markdown("#### 💡 AI Recommendations")
                for rec in insights['recommendations']:
                    st.markdown(f"• {rec}")
            
            if 'cost_optimization' in insights:
                st.markdown("#### 💰 Cost Optimization")
                st.write(insights['cost_optimization'])
    else:
        st.info("🤖 AI insights not available. Provide an Anthropic API key in the configuration to enable AI-powered analysis.")


def show_timeline_analysis_tab():
    """Show migration timeline analysis"""
    
    st.markdown("### 📅 Migration Timeline & Milestones")
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Timeline analysis not available. Please configure migration parameters first.")
        return
    
    params = st.session_state.migration_params
    timeline_weeks = params.get('migration_timeline_weeks', 12)
    
    # Create timeline phases
    phases = [
        {"phase": "Planning & Assessment", "weeks": max(2, timeline_weeks * 0.15), "description": "Requirements gathering, current state analysis"},
        {"phase": "Schema Migration", "weeks": max(2, timeline_weeks * 0.25), "description": "Convert schema, stored procedures, functions"},
        {"phase": "Data Migration", "weeks": max(3, timeline_weeks * 0.35), "description": "Initial load, incremental sync, validation"},
        {"phase": "Testing & Validation", "weeks": max(2, timeline_weeks * 0.15), "description": "Performance testing, data validation, UAT"},
        {"phase": "Go-Live & Support", "weeks": max(1, timeline_weeks * 0.10), "description": "Cutover, monitoring, post-migration support"}
    ]
    
    # Timeline visualization
    current_week = 0
    for i, phase in enumerate(phases):
        start_week = current_week
        end_week = current_week + phase["weeks"]
        
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            st.markdown(f"**{phase['phase']}**")
        
        with col2:
            st.markdown(f"Weeks {int(start_week)+1}-{int(end_week)}")
        
        with col3:
            st.markdown(phase['description'])
        
        current_week = end_week


def show_growth_analysis_dashboard():
    """Show comprehensive growth analysis dashboard"""
    
    st.markdown("### 📈 3-Year Growth Analysis & Projections")
    
    # Check if growth analysis exists
    if not hasattr(st.session_state, 'growth_analysis') or not st.session_state.growth_analysis:
        st.warning("⚠️ Growth analysis not available. Please run the analysis first.")
        
        # Show basic growth planning instead
        st.markdown("#### 🎯 Growth Planning Preview")
        st.info("""
        **Growth analysis will show:**
        - 3-year cost projections with growth factors
        - Resource scaling requirements
        - Seasonal peak planning
        - Cost optimization opportunities
        - Scaling recommendations by year
        
        Run the migration analysis to see detailed growth projections.
        """)
        return
    
    growth_analysis = st.session_state.growth_analysis
    growth_summary = growth_analysis['growth_summary']
    
    # Key Growth Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "3-Year Growth",
            f"{growth_summary['total_3_year_growth_percent']:.1f}%",
            delta=f"CAGR: {growth_summary['compound_annual_growth_rate']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Current Annual Cost",
            f"${growth_summary['year_0_cost']:,.0f}",
            delta="Baseline"
        )
    
    with col3:
        st.metric(
            "Year 3 Projected Cost",
            f"${growth_summary['year_3_cost']:,.0f}",
            delta=f"+${growth_summary['year_3_cost'] - growth_summary['year_0_cost']:,.0f}"
        )
    
    with col4:
        st.metric(
            "Total 3-Year Investment",
            f"${growth_summary['total_3_year_investment']:,.0f}",
            delta=f"Avg: ${growth_summary['average_annual_cost']:,.0f}/year"
        )
    
    # Growth Projection Charts
    st.markdown("#### 📊 Growth Projections")
    
    try:
        charts = create_growth_projection_charts(growth_analysis)
        for chart in charts:
            st.plotly_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating growth charts: {str(e)}")
        
        # Show basic growth table instead
        st.markdown("#### 📈 Year-by-Year Projections")
        projections = growth_analysis['yearly_projections']
        
        years_data = []
        for year in range(4):
            year_data = projections[f'year_{year}']
            years_data.append({
                'Year': f'Year {year}' if year > 0 else 'Current',
                'Monthly Cost': f"${year_data['total_monthly']:,.0f}",
                'Annual Cost': f"${year_data['total_annual']:,.0f}",
                'Peak Cost': f"${year_data['peak_annual']:,.0f}"
            })
        
        st.table(years_data)
    
    # Scaling Recommendations
    st.markdown("#### 🎯 Scaling Recommendations")
    
    recommendations = growth_analysis.get('scaling_recommendations', [])
    
    if recommendations:
        for rec in recommendations:
            priority_color = {
                'High': '#e53e3e',
                'Medium': '#d69e2e',
                'Low': '#38a169'
            }.get(rec['priority'], '#666666')
            
            st.markdown(f"""
            <div style="border-left: 4px solid {priority_color}; padding: 15px; margin: 10px 0; background: {priority_color}22;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: {priority_color};">{rec['type']} (Year {rec['year']})</strong><br>
                        {rec['description']}<br>
                        <em>Action: {rec['action']}</em>
                    </div>
                    <div style="text-align: right;">
                        <strong>Priority: {rec['priority']}</strong><br>
                        <span style="color: #38a169;">Potential Savings: ${rec['estimated_savings']:,.0f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical scaling issues identified in the 3-year projection.")
        
        # Show default recommendations
        st.markdown("#### 💡 General Growth Recommendations")
        default_recommendations = [
            "📊 **Monitor growth trends** - Track actual vs projected growth quarterly",
            "💰 **Reserved Instances** - Consider 1-3 year commitments for 30-40% savings",
            "🔄 **Auto-scaling** - Implement auto-scaling for variable workloads",
            "📈 **Capacity planning** - Review and adjust capacity every 6 months",
            "🗄️ **Data lifecycle** - Implement archiving for older data to reduce storage costs"
        ]
        
        for rec in default_recommendations:
            st.markdown(rec)


def create_growth_projection_charts(growth_analysis: Dict) -> List[go.Figure]:
    """Create comprehensive growth projection visualizations"""
    
    charts = []
    projections = growth_analysis['yearly_projections']
    
    # 1. 3-Year Cost Projection Chart
    years = ['Current', 'Year 1', 'Year 2', 'Year 3']
    annual_costs = [projections[f'year_{i}']['total_annual'] for i in range(4)]
    peak_costs = [projections[f'year_{i}']['peak_annual'] for i in range(4)]
    
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=years, y=annual_costs,
        mode='lines+markers',
        name='Average Annual Cost',
        line=dict(color='#3182ce', width=3),
        marker=dict(size=10)
    ))
    
    fig1.add_trace(go.Scatter(
        x=years, y=peak_costs,
        mode='lines+markers',
        name='Peak Annual Cost (with seasonality)',
        line=dict(color='#e53e3e', width=3, dash='dash'),
        marker=dict(size=10)
    ))
    
    fig1.update_layout(
        title='3-Year Cost Projection with Growth',
        xaxis_title='Timeline',
        yaxis_title='Annual Cost ($)',
        height=500,
        hovermode='x unified'
    )
    
    charts.append(fig1)
    
    # 2. Cost Component Growth Breakdown
    components = ['Compute', 'Storage', 'I/O', 'Backup', 'Monitoring']
    
    # Get first environment for component breakdown
    year_3_environments = projections['year_3']['environment_costs']
    if year_3_environments:
        year_3_env = list(year_3_environments.values())[0]
        
        component_costs = [
            year_3_env.get('instance_cost', 0) + year_3_env.get('reader_costs', 0),
            year_3_env.get('storage_cost', 0),
            year_3_env.get('iops_cost', 0),
            year_3_env.get('backup_cost', 0),
            year_3_env.get('monitoring_cost', 0)
        ]
        
        fig2 = go.Figure(data=[go.Pie(
            labels=components,
            values=component_costs,
            hole=0.4,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}',
            marker=dict(colors=['#3182ce', '#38a169', '#d69e2e', '#e53e3e', '#805ad5'])
        )])
        
        fig2.update_layout(
            title='Year 3 Cost Distribution by Component',
            height=500
        )
        
        charts.append(fig2)
    
    # 3. Resource Scaling Requirements
    environments = list(projections['year_0']['environment_costs'].keys())
    if environments:
        compute_scaling = []
        storage_scaling = []
        
        for env in environments:
            year_3_scaling = projections['year_3']['environment_costs'][env].get('resource_scaling', {})
            compute_scaling.append(year_3_scaling.get('compute_scaling', 1.0))
            storage_scaling.append(year_3_scaling.get('storage_scaling', 1.0))
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Bar(
            name='Compute Scaling',
            x=environments,
            y=compute_scaling,
            marker_color='#3182ce'
        ))
        
        fig3.add_trace(go.Bar(
            name='Storage Scaling',
            x=environments,
            y=storage_scaling,
            marker_color='#38a169'
        ))
        
        fig3.update_layout(
            title='3-Year Resource Scaling Requirements by Environment',
            xaxis_title='Environment',
            yaxis_title='Scaling Factor (1.0 = no scaling)',
            barmode='group',
            height=500
        )
        
        charts.append(fig3)
    
    return charts

def show_growth_analysis_dashboard():
    """Show comprehensive growth analysis dashboard"""
    
    st.markdown("### 📈 3-Year Growth Analysis & Projections")
    
    # Check if growth analysis exists
    if not hasattr(st.session_state, 'growth_analysis') or not st.session_state.growth_analysis:
        st.warning("⚠️ Growth analysis not available. Please run the analysis first.")
        
        # Show basic growth planning instead
        st.markdown("#### 🎯 Growth Planning Preview")
        st.info("""
        **Growth analysis will show:**
        - 3-year cost projections with growth factors
        - Resource scaling requirements
        - Seasonal peak planning
        - Cost optimization opportunities
        - Scaling recommendations by year
        
        Run the migration analysis to see detailed growth projections.
        """)
        return
    
    growth_analysis = st.session_state.growth_analysis
    growth_summary = growth_analysis['growth_summary']
    
    # Key Growth Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "3-Year Growth",
            f"{growth_summary['total_3_year_growth_percent']:.1f}%",
            delta=f"CAGR: {growth_summary['compound_annual_growth_rate']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Current Annual Cost",
            f"${growth_summary['year_0_cost']:,.0f}",
            delta="Baseline"
        )
    
    with col3:
        st.metric(
            "Year 3 Projected Cost",
            f"${growth_summary['year_3_cost']:,.0f}",
            delta=f"+${growth_summary['year_3_cost'] - growth_summary['year_0_cost']:,.0f}"
        )
    
    with col4:
        st.metric(
            "Total 3-Year Investment",
            f"${growth_summary['total_3_year_investment']:,.0f}",
            delta=f"Avg: ${growth_summary['average_annual_cost']:,.0f}/year"
        )
    
    # Growth Projection Charts
    st.markdown("#### 📊 Growth Projections")
    
    try:
        charts = create_growth_projection_charts(growth_analysis)
        
         # Display charts with unique keys - THIS IS THE KEY FIX
        for i, chart in enumerate(charts):
            st.plotly_chart(chart, use_container_width=True, key=f"growth_dashboard_chart_{i}")
    except Exception as e:
        st.error(f"Error creating growth charts: {str(e)}")
        
        # Show basic growth table instead
        st.markdown("#### 📈 Year-by-Year Projections")
        projections = growth_analysis['yearly_projections']
        
        years_data = []
        for year in range(4):
            year_data = projections[f'year_{year}']
            years_data.append({
                'Year': f'Year {year}' if year > 0 else 'Current',
                'Monthly Cost': f"${year_data['total_monthly']:,.0f}",
                'Annual Cost': f"${year_data['total_annual']:,.0f}",
                'Peak Cost': f"${year_data['peak_annual']:,.0f}"
            })
        
        st.table(years_data)
    
    # Scaling Recommendations
    st.markdown("#### 🎯 Scaling Recommendations")
    
    recommendations = growth_analysis.get('scaling_recommendations', [])
    
    if recommendations:
        for rec in recommendations:
            priority_color = {
                'High': '#e53e3e',
                'Medium': '#d69e2e',
                'Low': '#38a169'
            }.get(rec['priority'], '#666666')
            
            st.markdown(f"""
            <div style="border-left: 4px solid {priority_color}; padding: 15px; margin: 10px 0; background: {priority_color}22;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: {priority_color};">{rec['type']} (Year {rec['year']})</strong><br>
                        {rec['description']}<br>
                        <em>Action: {rec['action']}</em>
                    </div>
                    <div style="text-align: right;">
                        <strong>Priority: {rec['priority']}</strong><br>
                        <span style="color: #38a169;">Potential Savings: ${rec['estimated_savings']:,.0f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical scaling issues identified in the 3-year projection.")
        
        # Show default recommendations
        st.markdown("#### 💡 General Growth Recommendations")
        default_recommendations = [
            "📊 **Monitor growth trends** - Track actual vs projected growth quarterly",
            "💰 **Reserved Instances** - Consider 1-3 year commitments for 30-40% savings",
            "🔄 **Auto-scaling** - Implement auto-scaling for variable workloads",
            "📈 **Capacity planning** - Review and adjust capacity every 6 months",
            "🗄️ **Data lifecycle** - Implement archiving for older data to reduce storage costs"
        ]
        
        for rec in default_recommendations:
            st.markdown(rec)
            
def create_environment_comparison_chart(environment_costs: Dict) -> go.Figure:
    """Create environment cost comparison chart"""
    
    environments = list(environment_costs.keys())
    instance_costs = [env['instance_cost'] for env in environment_costs.values()]
    storage_costs = [env['storage_cost'] for env in environment_costs.values()]
    total_costs = [env['total_monthly'] for env in environment_costs.values()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Instance Cost',
        x=environments,
        y=instance_costs,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Storage Cost',
        x=environments,
        y=storage_costs,
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Monthly Cost by Environment',
        xaxis_title='Environment',
        yaxis_title='Monthly Cost ($)',
        barmode='stack',
        height=400
    )
    
    return fig

def create_migration_timeline_gantt(migration_params: Dict) -> go.Figure:
    """Create migration timeline Gantt chart"""
    
    timeline_weeks = migration_params.get('migration_timeline_weeks', 12)
    
    # Define migration phases
    phases = [
        {'name': 'Assessment & Planning', 'start': 0, 'duration': 2},
        {'name': 'Environment Setup', 'start': 1, 'duration': 3},
        {'name': 'Schema Migration', 'start': 3, 'duration': 2},
        {'name': 'Data Migration', 'start': 4, 'duration': 4},
        {'name': 'Application Testing', 'start': 6, 'duration': 3},
        {'name': 'User Acceptance Testing', 'start': 8, 'duration': 2},
        {'name': 'Go-Live Preparation', 'start': 9, 'duration': 2},
        {'name': 'Production Cutover', 'start': 11, 'duration': 1}
    ]
    
    # Create simple bar chart instead of Gantt to avoid dependencies
    fig = go.Figure()
    
    for i, phase in enumerate(phases):
        fig.add_trace(go.Bar(
            name=phase['name'],
            x=[phase['name']],
            y=[phase['duration']],
            text=f"{phase['duration']} weeks",
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Migration Phase Duration',
        xaxis_title='Phase',
        yaxis_title='Duration (weeks)',
        showlegend=False,
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

# ===========================
# PDF REPORT GENERATION
# ===========================

def generate_executive_summary_pdf(analysis_results: Dict, migration_params: Dict) -> io.BytesIO:
    """Generate executive summary PDF report"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#2d3748'),
        alignment=1
    )
    
    story = []
    
    # Title and header
    story.append(Paragraph("AWS Database Migration Analysis", title_style))
    story.append(Paragraph("Executive Summary Report", styles['Heading2']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Key metrics table
    cost_analysis = analysis_results
    migration_costs = cost_analysis['migration_costs']
    
    metrics_data = [
        ['Metric', 'Value', 'Impact'],
        ['Monthly AWS Cost', f"${cost_analysis['monthly_aws_cost']:,.0f}", 'Operational'],
        ['Annual AWS Cost', f"${cost_analysis['annual_aws_cost']:,.0f}", 'Budget Planning'],
        ['Migration Investment', f"${migration_costs['total']:,.0f}", 'One-time'],
        ['ROI Timeline', '18-24 months', 'Financial'],
        ['Risk Level', 'Medium', 'Manageable']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Key Recommendations", styles['Heading2']))
    
    recommendations = [
        "Proceed with phased migration approach starting with non-production environments",
        "Implement AWS DMS for initial data synchronization",
        "Plan for 12-16 week migration timeline with 2-week buffer",
        "Establish comprehensive testing protocols for each environment",
        "Consider Aurora for production workloads to optimize performance and cost"
    ]
    
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 6))
    
    # Next steps
    story.append(Spacer(1, 20))
    story.append(Paragraph("Next Steps", styles['Heading2']))
    
    next_steps = [
        "1. Obtain stakeholder approval and budget allocation",
        "2. Form migration team and assign roles",
        "3. Set up AWS infrastructure and migration tools",
        "4. Begin with development environment migration",
        "5. Execute production cutover after successful testing"
    ]
    
    for step in next_steps:
        story.append(Paragraph(step, styles['Normal']))
        story.append(Spacer(1, 6))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_technical_report_pdf(analysis_results: Dict, recommendations: Dict, migration_params: Dict) -> io.BytesIO:
    """Generate detailed technical PDF report"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("AWS Database Migration - Technical Report", styles['Title']))
    story.append(Spacer(1, 20))
    
    # Migration parameters
    story.append(Paragraph("Migration Configuration", styles['Heading2']))
    
    config_data = [
        ['Parameter', 'Value'],
        ['Source Engine', migration_params.get('source_engine', 'N/A')],
        ['Target Engine', migration_params.get('target_engine', 'N/A')],
        ['Data Size', f"{migration_params.get('data_size_gb', 0):,} GB"],
        ['Timeline', f"{migration_params.get('migration_timeline_weeks', 0)} weeks"],
        ['Environments', str(len(recommendations))],
        ['Migration Type', DatabaseEngine.get_migration_type(
            migration_params.get('source_engine', ''),
            migration_params.get('target_engine', '')
        )]
    ]
    
    config_table = Table(config_data, colWidths=[2.5*inch, 2*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    story.append(config_table)
    story.append(PageBreak())
    
    # Environment recommendations
    story.append(Paragraph("Environment Recommendations", styles['Heading2']))
    
    env_data = [['Environment', 'Instance Class', 'CPU/RAM', 'Storage', 'Multi-AZ']]
    
    for env_name, rec in recommendations.items():
        env_data.append([
            env_name,
            rec['instance_class'],
            f"{rec['cpu_cores']} vCPU / {rec['ram_gb']} GB",
            f"{rec['storage_gb']} GB",
            'Yes' if rec['multi_az'] else 'No'
        ])
    
    env_table = Table(env_data, colWidths=[1.2*inch, 1.3*inch, 1.2*inch, 1*inch, 0.8*inch])
    env_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    
    story.append(env_table)
    story.append(Spacer(1, 20))
    
    # Cost breakdown
    story.append(Paragraph("Cost Analysis", styles['Heading2']))
    
    env_costs = analysis_results['environment_costs']
    cost_data = [['Environment', 'Instance', 'Storage', 'Backup', 'Total Monthly']]
    
    for env_name, costs in env_costs.items():
        cost_data.append([
            env_name,
            f"${costs['instance_cost']:,.0f}",
            f"${costs['storage_cost']:,.0f}",
            f"${costs['backup_cost']:,.0f}",
            f"${costs['total_monthly']:,.0f}"
        ])
    
    cost_table = Table(cost_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1*inch, 1.2*inch])
    cost_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e53e3e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    
    story.append(cost_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ===========================
# STREAMLIT APPLICATION
# ===========================

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'environment_specs': {},
        'migration_params': {},
        'network_analysis': None,        # <-- ADD THIS LINE
        'transfer_analysis': None,       # <-- ADD THIS LINE
        
        'analysis_results': None,
        'recommendations': None,
        'risk_assessment': None,
        'ai_insights': None,
        # ADD THESE NEW LINES:
        'enhanced_recommendations': None,
        'enhanced_analysis_results': None,
        'enhanced_cost_chart': None,
        'growth_analysis': None,  # ADD THIS LINE
        'growth_projections': None,  # ADD THIS LINE
          # ADD THESE NEW LINES:
        'cost_reconciliation_data': None,
        'aws_actual_costs': None,
        'reconciliation_results': None,
        'cost_validation_data': None,     
        'enhanced_cost_chart': None,
        'growth_analysis': None,
        'growth_projections': None,
        'optimization_results': None, 
        'reconciliation_settings': {}# <-- ADD THIS LINE
    
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            
    # ADD these new vROps session state variables
        if 'vrops_servers' not in st.session_state:
            st.session_state.vrops_servers = []
        if 'vrops_analysis' not in st.session_state:
            st.session_state.vrops_analysis = {}
        if 'vrops_metrics' not in st.session_state:
            st.session_state.vrops_metrics = {}
        if 'vrops_database_servers' not in st.session_state:
            st.session_state.vrops_database_servers = []
        if 'vrops_connection_info' not in st.session_state:
            st.session_state.vrops_connection_info = {}
            

def test_claude_ai_connection():
    """Test Claude AI integration"""
    
    st.markdown("### 🧪 Test Claude AI Connection")
    
    api_key = st.text_input("Enter Anthropic API Key:", type="password")
    
    if st.button("🤖 Test Claude AI"):
        if not api_key:
            st.error("Please enter an API key")
            return
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            with st.spinner("Testing Claude AI..."):
                message = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=100,
                    messages=[{
                        "role": "user",
                        "content": "This is a test. Please respond: 'Claude AI is working correctly!'"
                    }]
                )
                
                response = message.content[0].text
                
                if "working correctly" in response.lower():
                    st.success("✅ Claude AI is working!")
                    st.info(f"Response: {response}")
                else:
                    st.warning("⚠️ Unexpected response")
                    st.write(response)
                    
        except ImportError:
            st.error("❌ Run: pip install anthropic")
        except Exception as e:
            st.error(f"❌ Test failed: {str(e)}")



def main():
    """Main Streamlit application"""
    
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="enterprise-header">
        <h1>🚀 Enterprise AWS Database Migration Tool</h1>
        <p>AI-Powered Analysis • Real-time AWS Pricing • Comprehensive Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.radio(
            "Select Section:",
            [
                "🔧 Migration Configuration",
                "📊 Environment Setup",
                "🌐 Network Analysis",
                "🧠 AI Optimizer",
                "🚀 Analysis & Recommendations",
                "📈 Results Dashboard",
                "🔄 Cost Reconciliation",  # ADD THIS LINE
                "📄 Reports & Export",
                "📊 Enhanced Recommendations",
                "📡 vROps Integration"  # <-- ADD THIS NEW OPTION
            ]
        )
        
        # Enhanced Status Section
        st.markdown("---")
        st.markdown("## 📋 Status")
        
        # Environment Configuration Status
        env_specs = getattr(st.session_state, 'environment_specs', {})
        if env_specs and len(env_specs) > 0:
            st.success(f"✅ {len(env_specs)} environments configured")
            st.caption(f"Num environments: `{len(env_specs)}`")
            
            # Check if enhanced data
            is_enhanced = is_enhanced_environment_data(env_specs)
            st.caption(f"Enhanced data: `{is_enhanced}`")
        else:
            st.warning("⚠️ Configure environments")
        
        # Migration Parameters Status
        if st.session_state.migration_params:
            st.success("✅ Migration parameters set")
        else:
            st.warning("⚠️ Set migration parameters")
        
        # Analysis Results Status
        has_regular_results = st.session_state.analysis_results is not None
        has_enhanced_results = hasattr(st.session_state, 'enhanced_analysis_results') and st.session_state.enhanced_analysis_results is not None
        
        if has_regular_results or has_enhanced_results:
            st.success("✅ Analysis complete")
            
            # Show cost metrics
            if has_enhanced_results:
                results = st.session_state.enhanced_analysis_results
                st.metric("Monthly Cost", f"${results['monthly_aws_cost']:,.0f}")
                migration_cost = results['migration_costs']['total']
                st.metric("Migration Cost", f"${migration_cost:,.0f}")
                st.info("🔬 Enhanced Analysis")
            elif has_regular_results:
                results = st.session_state.analysis_results
                st.metric("Monthly Cost", f"${results['monthly_aws_cost']:,.0f}")
                migration_cost = results.get('migration_costs', {}).get('total', 0)
                st.metric("Migration Cost", f"${migration_cost:,.0f}")
                st.info("📊 Standard Analysis")
        else:
            st.info("ℹ️ Analysis pending")
        
        # Network Analysis Status
        if hasattr(st.session_state, 'transfer_analysis') and st.session_state.transfer_analysis:
            st.success("✅ Network analysis complete")
            recommendations = st.session_state.transfer_analysis.get('recommendations', {})
            primary = recommendations.get('primary_recommendation', {})
            if primary:
                st.caption("**Recommended Pattern:**")
                st.caption(primary.get('pattern_name', 'N/A'))
        else:
            st.info("ℹ️ Network analysis pending")
        
        # AI Optimizer Status
        if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
            st.success("✅ AI optimization complete")
            
            # Show optimization metrics
            total_monthly = sum([env['cost_analysis']['monthly_breakdown']['total'] 
                               for env in st.session_state.optimization_results.values()])
            avg_score = sum([env['optimization_score'] 
                           for env in st.session_state.optimization_results.values()]) / len(st.session_state.optimization_results)
            
            st.metric("Optimized Monthly Cost", f"${total_monthly:,.0f}")
            st.metric("Avg Optimization Score", f"{avg_score:.1f}/100")
        else:
            st.info("ℹ️ AI optimization pending")
        
        # Growth Analysis Status (if available)
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            st.success("✅ Growth analysis complete")
            growth_summary = st.session_state.growth_analysis['growth_summary']
            growth_percent = growth_summary['total_3_year_growth_percent']
            st.metric("3-Year Growth", f"{growth_percent:.1f}%")
        
        # Quick Actions
        st.markdown("---")
        st.markdown("## ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.experimental_rerun()
        
        if st.session_state.migration_params and st.session_state.environment_specs:
            if st.button("🚀 Run Analysis", use_container_width=True):
                # Switch to analysis page
                st.session_state.page = "🚀 Analysis & Recommendations"
                st.experimental_rerun()
        
        # System Info
        st.markdown("---")
        st.markdown("## ℹ️ System Info")
        st.caption(f"**Session:** {len(st.session_state)} variables")
        st.caption(f"**Generated:** {datetime.now().strftime('%H:%M:%S')}")
    
        # Main content area routing remains the same
        if page == "🔧 Migration Configuration":
            show_migration_configuration()
        elif page == "📊 Environment Setup":
            show_enhanced_environment_setup_with_cluster_config()
        elif page == "🌐 Network Analysis":
            show_network_transfer_analysis()
        elif page == "🧠 AI Optimizer":
            show_optimized_recommendations()
        elif page == "🚀 Analysis & Recommendations":
            show_analysis_section_fixed()
        elif page == "📈 Results Dashboard":
            show_results_dashboard()
        elif page == "🔄 Cost Reconciliation":  # ADD THIS BLOCK
            show_cost_reconciliation_page()
        elif page == "📄 Reports & Export":
            show_reports_section()
        elif page == "📊 Enhanced Recommendations":
            show_enhanced_recommendations_dashboard()
        elif page == "📡 vROps Integration":
            show_vrops_main_page()  # ✅ ADD THIS LINE - you were missing the function call!
        

            
def show_migration_configuration():
    """Show migration configuration interface with growth planning"""
    
    st.markdown("## 🔧 Migration Configuration")
    
    # Source and target engine selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Source Database")
        source_engine = st.selectbox(
            "Source Engine",
            options=list(DatabaseEngine.ENGINES.keys()),
            format_func=lambda x: DatabaseEngine.ENGINES[x]['name'],
            key="source_engine"
        )
        
        if source_engine:
            st.info(f"**Features:** {', '.join(DatabaseEngine.ENGINES[source_engine]['features'])}")
    
    with col2:
        st.markdown("### 📤 Target AWS Database")
        
        if source_engine:
            target_options = DatabaseEngine.ENGINES[source_engine]['aws_targets']
            target_engine = st.selectbox(
                "Target Engine",
                options=target_options,
                format_func=lambda x: DatabaseEngine.ENGINES.get(x, {'name': x.title()})['name'] if x in DatabaseEngine.ENGINES else x.replace('-', ' ').title(),
                key="target_engine"
            )
        if source_engine:
            st.info(f"**Features:** {', '.join(DatabaseEngine.ENGINES[source_engine]['features'])}")
    
    # Migration parameters
    st.markdown("### ⚙️ Migration Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💾 Data Configuration")
        data_size_gb = st.number_input("Total Data Size (GB)", min_value=1, max_value=100000, value=1000)
        num_applications = st.number_input("Connected Applications", min_value=1, max_value=50, value=3)
        num_stored_procedures = st.number_input("Stored Procedures/Functions", min_value=0, max_value=10000, value=50)
    
    with col2:
        st.markdown("#### ⏱️ Timeline & Resources")
        migration_timeline_weeks = st.slider("Migration Timeline (weeks)", min_value=4, max_value=52, value=12)
        team_size = st.number_input("Team Size", min_value=2, max_value=20, value=5)
        team_expertise = st.selectbox("Team Expertise Level", ["low", "medium", "high"], index=1)
    
    with col3:
        st.markdown("#### 🌐 Infrastructure")
        region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"], index=0)
        use_direct_connect = st.checkbox("Use AWS Direct Connect", value=True)
        bandwidth_mbps = st.selectbox("Bandwidth (Mbps)", [100, 1000, 10000], index=1)
        migration_budget = st.number_input("Migration Budget ($)", min_value=10000, max_value=5000000, value=500000)

    # ADD NEW GROWTH PLANNING SECTION:
    st.markdown("### 📈 Growth Planning & Forecasting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Workload Growth")
        annual_data_growth = st.slider(
            "Annual Data Growth Rate (%)", 
            min_value=0, max_value=100, value=15,
            help="Expected annual growth in database size"
        )
        
        annual_user_growth = st.slider(
            "Annual User Growth Rate (%)", 
            min_value=0, max_value=200, value=25,
            help="Expected annual growth in concurrent users"
        )
        
        annual_transaction_growth = st.slider(
            "Annual Transaction Growth Rate (%)", 
            min_value=0, max_value=150, value=20,
            help="Expected annual growth in transaction volume"
        )
    
    with col2:
        st.markdown("#### 🎯 Growth Scenarios")
        growth_scenario = st.selectbox(
            "Growth Scenario",
            ["Conservative", "Moderate", "Aggressive", "Custom"],
            index=1,
            help="Predefined growth scenarios or custom settings"
        )
        
        if growth_scenario != "Custom":
            # Auto-adjust growth rates based on scenario
            scenario_multipliers = {
                "Conservative": 0.7,
                "Moderate": 1.0,
                "Aggressive": 1.5
            }
            multiplier = scenario_multipliers[growth_scenario]
            annual_data_growth = int(annual_data_growth * multiplier)
            annual_user_growth = int(annual_user_growth * multiplier)
            annual_transaction_growth = int(annual_transaction_growth * multiplier)
            
            st.info(f"**{growth_scenario} Scenario Applied:**")
            st.write(f"Data Growth: {annual_data_growth}%")
            st.write(f"User Growth: {annual_user_growth}%")
            st.write(f"Transaction Growth: {annual_transaction_growth}%")
        
        seasonality_factor = st.slider(
            "Seasonality Factor", 
            min_value=1.0, max_value=3.0, value=1.2, step=0.1,
            help="Peak season multiplier (1.0 = no seasonality)"
        )
        
        scaling_strategy = st.selectbox(
            "Scaling Strategy",
            ["Auto-scaling", "Manual scaling", "Over-provision"],
            help="How to handle growth in infrastructure"
        )

    # AI Configuration
    st.markdown("### 🤖 AI Integration")
    
    anthropic_api_key = st.text_input(
        "Anthropic API Key (Optional)",
        type="password",
        help="Provide your Anthropic API key for AI-powered insights"
    )
    
    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        st.session_state.migration_params = {
            'source_engine': source_engine,
            'target_engine': target_engine,
            'data_size_gb': data_size_gb,
            'num_applications': num_applications,
            'num_stored_procedures': num_stored_procedures,
            'migration_timeline_weeks': migration_timeline_weeks,
            'team_size': team_size,
            'team_expertise': team_expertise,
            'region': region,
            'use_direct_connect': use_direct_connect,
            'bandwidth_mbps': bandwidth_mbps,
            'migration_budget': migration_budget,
            'anthropic_api_key': anthropic_api_key,
            'estimated_migration_cost': 0,
            # ADD THESE NEW GROWTH PARAMETERS:
            'annual_data_growth': annual_data_growth,
            'annual_user_growth': annual_user_growth,
            'annual_transaction_growth': annual_transaction_growth,
            'growth_scenario': growth_scenario,
            'seasonality_factor': seasonality_factor,
            'scaling_strategy': scaling_strategy
        }
        
        st.success("✅ Configuration saved! Proceed to Environment Setup.")
        st.balloons()

def show_environment_analysis():
    """Show environment analysis dashboard"""
    
    st.markdown("### 🏢 Environment Analysis")
    
    # Check for enhanced recommendations first
    if hasattr(st.session_state, 'enhanced_recommendations') and st.session_state.enhanced_recommendations:
        show_enhanced_environment_analysis()
        return
    
    if not hasattr(st.session_state, 'recommendations') or not st.session_state.recommendations:
        st.warning("Environment analysis not available. Please run the analysis first.")
        return
    
    recommendations = st.session_state.recommendations
    environment_specs = st.session_state.environment_specs
    
    if not environment_specs:
        st.warning("No environment specifications available.")
        return
    
    # Environment comparison
    env_comparison_data = []
    
    for env_name, rec in recommendations.items():
        specs = environment_specs.get(env_name, {})
        
        env_comparison_data.append({
            'Environment': env_name,
            'Type': rec.get('environment_type', 'Unknown').title(),
            'Current Resources': f"{specs.get('cpu_cores', 'N/A')} cores, {specs.get('ram_gb', 'N/A')} GB RAM",
            'Recommended Instance': rec.get('instance_class', 'N/A'),
            'Storage': f"{specs.get('storage_gb', 'N/A')} GB",
            'Multi-AZ': 'Yes' if rec.get('multi_az', False) else 'No',
            'Daily Usage': f"{specs.get('daily_usage_hours', 'N/A')} hours"
        })
    
    if env_comparison_data:
        env_df = pd.DataFrame(env_comparison_data)
        st.dataframe(env_df, use_container_width=True)
    else:
        st.warning("No environment comparison data available.")
    
    # Environment-specific insights
    st.markdown("#### 💡 Environment Insights")
    
    for env_name, rec in recommendations.items():
        with st.expander(f"🔍 {env_name} Environment Details"):
            specs = environment_specs.get(env_name, {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Current Configuration**")
                st.write(f"CPU Cores: {specs.get('cpu_cores', 'N/A')}")
                st.write(f"RAM: {specs.get('ram_gb', 'N/A')} GB")
                st.write(f"Storage: {specs.get('storage_gb', 'N/A')} GB")
                st.write(f"Daily Usage: {specs.get('daily_usage_hours', 'N/A')} hours")
            
            with col2:
                st.markdown("**AWS Recommendation**")
                st.write(f"Instance: {rec.get('instance_class', 'N/A')}")
                st.write(f"Environment Type: {rec.get('environment_type', 'Unknown').title()}")
                st.write(f"Multi-AZ: {'Yes' if rec.get('multi_az', False) else 'No'}")
                
                # Get instance specs (simplified)
                instance_class = rec.get('instance_class', '')
                if 'xlarge' in instance_class:
                    aws_cpu = 16 if '4xlarge' in instance_class else 8 if '2xlarge' in instance_class else 4
                    aws_ram = aws_cpu * 8
                elif 'large' in instance_class:
                    aws_cpu = 2
                    aws_ram = 8
                else:
                    aws_cpu = 1
                    aws_ram = 4
                
                st.write(f"AWS vCPUs: {aws_cpu}")
                st.write(f"AWS RAM: {aws_ram} GB")
            
            with col3:
                st.markdown("**Optimization Notes**")
                
                # Generate optimization suggestions
                env_type = rec.get('environment_type', '')
                if env_type == 'production':
                    st.write("✅ Production-grade configuration")
                    st.write("✅ Multi-AZ for high availability")
                elif env_type == 'development':
                    st.write("💡 Cost-optimized for development")
                    st.write("💡 Single-AZ to reduce costs")
                
                daily_hours = specs.get('daily_usage_hours', 24)
                if daily_hours < 12:
                    st.write("⚡ Consider Aurora Serverless for variable workloads")





def show_bulk_upload_interface():
    """Show bulk upload interface for environments"""
    
    st.markdown("### 📁 Bulk Environment Upload")
    
    # Sample template
    with st.expander("📋 Download Sample Template", expanded=False):
        sample_data = {
            'environment': ['Development', 'QA', 'SQA', 'Production'],
            'cpu_cores': [4, 8, 16, 32],
            'ram_gb': [16, 32, 64, 128],
            'storage_gb': [100, 500, 1000, 2000],
            'daily_usage_hours': [8, 12, 16, 24],
            'peak_connections': [20, 50, 100, 500]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)
        
        csv_data = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_data,
            file_name="environment_template.csv",
            mime="text/csv"
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload a CSV file with environment specifications"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['environment', 'cpu_cores', 'ram_gb', 'storage_gb']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return
            
            # Process environments
            environment_specs = {}
            for _, row in df.iterrows():
                env_name = str(row['environment']).strip()
                environment_specs[env_name] = {
                    'cpu_cores': int(row['cpu_cores']),
                    'ram_gb': int(row['ram_gb']),
                    'storage_gb': int(row['storage_gb']),
                    'daily_usage_hours': int(row.get('daily_usage_hours', 24)),
                    'peak_connections': int(row.get('peak_connections', 100))
                }
            
            st.session_state.environment_specs = environment_specs
            st.success(f"✅ Successfully loaded {len(environment_specs)} environments!")
            
            # Display loaded data
            st.markdown("#### 📊 Loaded Environments")
            display_df = pd.DataFrame.from_dict(environment_specs, orient='index')
            st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_manual_environment_setup():
    """Show manual environment setup interface"""
    
    st.markdown("### 📝 Manual Environment Configuration")
    
    # Number of environments
    num_environments = st.number_input("Number of Environments", min_value=1, max_value=10, value=4)
    
    # Environment configuration
    environment_specs = {}
    default_names = ['Development', 'QA', 'SQA', 'Production']
    
    cols = st.columns(min(num_environments, 3))
    
    for i in range(num_environments):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            with st.expander(f"🏢 Environment {i+1}", expanded=True):
                env_name = st.text_input(
                    "Environment Name",
                    value=default_names[i] if i < len(default_names) else f"Environment_{i+1}",
                    key=f"env_name_{i}"
                )
                
                cpu_cores = st.number_input(
                    "CPU Cores",
                    min_value=1, max_value=128,
                    value=[4, 8, 16, 32][min(i, 3)],
                    key=f"cpu_{i}"
                )
                
                ram_gb = st.number_input(
                    "RAM (GB)",
                    min_value=4, max_value=1024,
                    value=[16, 32, 64, 128][min(i, 3)],
                    key=f"ram_{i}"
                )
                
                storage_gb = st.number_input(
                    "Storage (GB)",
                    min_value=20, max_value=50000,
                    value=[100, 500, 1000, 2000][min(i, 3)],
                    key=f"storage_{i}"
                )
                
                daily_usage_hours = st.slider(
                    "Daily Usage (Hours)",
                    min_value=1, max_value=24,
                    value=[8, 12, 16, 24][min(i, 3)],
                    key=f"usage_{i}"
                )
                
                peak_connections = st.number_input(
                    "Peak Connections",
                    min_value=1, max_value=10000,
                    value=[20, 50, 100, 500][min(i, 3)],
                    key=f"connections_{i}"
                )
                
                environment_specs[env_name] = {
                    'cpu_cores': cpu_cores,
                    'ram_gb': ram_gb,
                    'storage_gb': storage_gb,
                    'daily_usage_hours': daily_usage_hours,
                    'peak_connections': peak_connections
                }
    
    if st.button("💾 Save Environment Configuration", type="primary", use_container_width=True):
        st.session_state.environment_specs = environment_specs
        st.success("✅ Environment configuration saved!")
        
        # Display summary
        st.markdown("#### 📊 Configuration Summary")
        summary_df = pd.DataFrame.from_dict(environment_specs, orient='index')
        st.dataframe(summary_df, use_container_width=True)

def show_analysis_section_fixed():
    """Show analysis and recommendations section - UPDATED"""
    
    st.markdown("## 🚀 Migration Analysis & Recommendations")
    
    # Check prerequisites
    if not st.session_state.migration_params:
        st.error("❌ Migration configuration required")
        st.info("👆 Please complete the 'Migration Configuration' section first")
        return
    
    if not st.session_state.environment_specs:
        st.error("❌ Environment configuration required")
        st.info("👆 Please complete the 'Environment Setup' section first")
        return
    
    # Display current configuration
    st.markdown("### 📋 Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        params = st.session_state.migration_params
        st.markdown(f"""
        **Migration Type:** {params['source_engine']} → {params['target_engine']}  
        **Data Size:** {params['data_size_gb']:,} GB  
        **Timeline:** {params['migration_timeline_weeks']} weeks  
        **Team Size:** {params['team_size']} members  
        **Budget:** ${params['migration_budget']:,}
        """)
    
    with col2:
        envs = st.session_state.environment_specs
        st.markdown(f"**Environments:** {len(envs)}")
        
        # Show first few environments
        count = 0
        for env_name, specs in envs.items():
            if count < 4:  # Show max 4 environments
                cpu_cores = specs.get('cpu_cores', 'N/A')
                ram_gb = specs.get('ram_gb', 'N/A')
                st.markdown(f"• **{env_name}:** {cpu_cores} cores, {ram_gb} GB RAM")
                count += 1
        
        if len(envs) > 4:
            st.markdown(f"• ... and {len(envs) - 4} more environments")
    
    # Detect configuration type
    is_enhanced = is_enhanced_environment_data(st.session_state.environment_specs)
    
    if is_enhanced:
        st.info("🔬 Enhanced cluster configuration detected - Writer/Reader analysis will be performed")
    else:
        st.info("📊 Standard configuration detected - Basic RDS analysis will be performed")
    
    # Run analysis - USE THE FIXED FUNCTION
    if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
        # Clear any previous results
        st.session_state.analysis_results = None
        if hasattr(st.session_state, 'enhanced_analysis_results'):
            st.session_state.enhanced_analysis_results = None
        
        with st.spinner("🔄 Analyzing migration requirements..."):
            run_streamlit_migration_analysis()  # Use the fixed function

# ADD THIS FUNCTION to your streamlit_app.py file:

def run_streamlit_migration_analysis():
    """Run migration analysis with growth projections - ENHANCED VERSION"""
    
    try:
        # Initialize analyzer
        anthropic_api_key = st.session_state.migration_params.get('anthropic_api_key')
        analyzer = MigrationAnalyzer(anthropic_api_key)
        
        # Step 1: Calculate recommendations
        st.write("📊 Calculating instance recommendations...")
        recommendations = analyzer.calculate_instance_recommendations(st.session_state.environment_specs)
        st.session_state.recommendations = recommendations
        
        # Step 2: Calculate costs
        st.write("💰 Analyzing costs...")
        cost_analysis = analyzer.calculate_migration_costs(recommendations, st.session_state.migration_params)
        st.session_state.analysis_results = cost_analysis
        
        # Step 3: Risk assessment
        st.write("⚠️ Assessing risks...")
        risk_assessment = create_default_risk_assessment()
        st.session_state.risk_assessment = risk_assessment
        
        # Step 4: Growth Analysis (NEW)
        st.write("📈 Calculating 3-year growth projections...")
        growth_analyzer = GrowthAwareCostAnalyzer()
        growth_analysis = growth_analyzer.calculate_3_year_growth_projection(
            cost_analysis, st.session_state.migration_params
        )
        st.session_state.growth_analysis = growth_analysis
        
        # Step 5: AI insights
        if anthropic_api_key:
            st.write("🤖 Generating AI insights...")
            try:
                ai_insights = asyncio.run(analyzer.generate_ai_insights(cost_analysis, st.session_state.migration_params))
                st.session_state.ai_insights = ai_insights
                st.success("✅ AI insights generated")
            except Exception as e:
                st.warning(f"AI insights failed: {str(e)}")
                st.session_state.ai_insights = {
                    'summary': f"Migration analysis complete. Monthly cost: ${cost_analysis['monthly_aws_cost']:,.0f}",
                    'error': str(e)
                }
        else:
            st.info("ℹ️ Provide Anthropic API key for AI insights")
        
        st.success("✅ Analysis complete with growth projections!")
        
        # Show enhanced summary
        show_analysis_summary_with_growth()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        create_basic_fallback()

def show_analysis_summary_with_growth():
    """Enhanced analysis summary including growth metrics"""
    
    st.markdown("#### 🎯 Analysis Summary with Growth Projections")
    
    col1, col2, col3, col4 = st.columns(4)
    
    results = st.session_state.analysis_results
    
    with col1:
        st.metric("Current Monthly Cost", f"${results['monthly_aws_cost']:,.0f}")
    
    with col2:
        migration_cost = results.get('migration_costs', {}).get('total', 0)
        st.metric("Migration Cost", f"${migration_cost:,.0f}")
    
    with col3:
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            growth_percent = st.session_state.growth_analysis['growth_summary']['total_3_year_growth_percent']
            st.metric("3-Year Growth", f"{growth_percent:.1f}%")
        else:
            st.metric("3-Year Growth", "Calculating...")
    
    with col4:
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            total_investment = st.session_state.growth_analysis['growth_summary']['total_3_year_investment']
            st.metric("3-Year Investment", f"${total_investment:,.0f}")
        else:
            if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
                risk_level = st.session_state.risk_assessment['risk_level']['level']
                st.metric("Risk Level", risk_level)
    
    st.info("📈 View detailed growth projections in the 'Results Dashboard' section")

def show_simple_summary():
    """Show simple analysis summary"""
    
    st.markdown("#### 🎯 Analysis Summary")
    col1, col2, col3 = st.columns(3)
    
    results = st.session_state.analysis_results
    
    if results:
        with col1:
            st.metric("Monthly Cost", f"${results['monthly_aws_cost']:,.0f}")
        
        with col2:
            migration_cost = results.get('migration_costs', {}).get('total', 0)
            st.metric("Migration Cost", f"${migration_cost:,.0f}")
        
        with col3:
            if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
                risk_level = st.session_state.risk_assessment['risk_level']['level']
                st.metric("Risk Level", risk_level)
    
    st.info("📈 View detailed results in the 'Results Dashboard' section")


def create_basic_fallback():
    """Create basic fallback when analysis completely fails"""
    
    st.warning("⚠️ Creating fallback analysis...")
    
    # Basic recommendations
    recommendations = {}
    total_cost = 0
    
    for env_name, specs in st.session_state.environment_specs.items():
        recommendations[env_name] = {
            'environment_type': 'production' if 'prod' in env_name.lower() else 'development',
            'instance_class': 'db.r5.large',
            'cpu_cores': specs.get('cpu_cores', 4),
            'ram_gb': specs.get('ram_gb', 16),
            'storage_gb': specs.get('storage_gb', 500),
            'multi_az': 'prod' in env_name.lower()
        }
        total_cost += 500  # $500 per environment estimate
    
    st.session_state.recommendations = recommendations
    
    # Basic cost analysis
    st.session_state.analysis_results = {
        'monthly_aws_cost': total_cost,
        'annual_aws_cost': total_cost * 12,
        'environment_costs': {env: {'total_monthly': 500, 'instance_cost': 400, 'storage_cost': 100} 
                            for env in recommendations.keys()},
        'migration_costs': {'total': 50000, 'dms_instance': 20000, 'data_transfer': 10000, 'professional_services': 20000}
    }
    
    # Basic risk assessment
    st.session_state.risk_assessment = get_fallback_risk_assessment()
    
    st.success("✅ Fallback analysis created")
                
def run_enhanced_migration_analysis():
    """Run enhanced migration analysis with Writer/Reader support"""
    
    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedMigrationAnalyzer()
        
         # Step 1: Calculate enhanced recommendations
        st.write("📊 Calculating cluster recommendations...")
        recommendations = analyzer.calculate_enhanced_instance_recommendations(st.session_state.environment_specs)
        st.session_state.enhanced_recommendations = recommendations
        
        # Step 2: Calculate enhanced costs
        st.write("💰 Analyzing cluster costs...")
        cost_analysis = analyzer.calculate_enhanced_migration_costs(recommendations, st.session_state.migration_params)
        st.session_state.enhanced_analysis_results = cost_analysis
        
        # Step 3: Generate risk assessment using enhanced data
        st.write("⚠️ Assessing risks...")
        risk_assessment = calculate_migration_risks(st.session_state.migration_params, recommendations)
        st.session_state.risk_assessment = risk_assessment
        
        # Step 4: Generate cost comparison
        st.write("📈 Generating cost comparisons...")
        generate_enhanced_cost_visualizations()
        
        st.success("✅ Enhanced cluster analysis complete!")
        
        # Show summary
        show_enhanced_analysis_summary()
        
        # Provide navigation hint
        st.info("📈 View detailed results in the 'Results Dashboard' section")
        
    except Exception as e:
        st.error(f"❌ Enhanced analysis failed: {str(e)}")
        st.code(str(e))
        
        # Provide troubleshooting info
        st.markdown("### 🔧 Troubleshooting")
        st.markdown("If the error persists:")
        st.markdown("1. Check that all environment fields are properly filled")
        st.markdown("2. Verify that numerical values are within valid ranges")
        st.markdown("3. Try using the 'Simple Configuration' option instead")

def show_results_dashboard():
    """Show comprehensive results dashboard with all analysis views"""
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ No analysis results available. Please run the migration analysis first.")
        return
    
    st.markdown("## 📊 Migration Analysis Results")
    
    # Check for enhanced results
    has_enhanced_results = (
        hasattr(st.session_state, 'enhanced_analysis_results') and 
        st.session_state.enhanced_analysis_results is not None
    )
    
    # FIXED: Match the number of variables with the number of tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,tab9 = st.tabs([
        "💰 Cost Summary",
        "📈 Growth Projections",
        "💎 Enhanced Analysis",
        "⚠️ Risk Assessment", 
        "🏢 Environment Analysis",
        "📊 Visualizations",
        "🤖 AI Insights",
        "📅 Timeline"
        "📊 Enhanced Recommendations"
    ])
    
    with tab1:
        show_basic_cost_summary()
    
    with tab2:
        show_growth_analysis_dashboard()
    
    with tab3:
        if has_enhanced_results:
            show_enhanced_cost_analysis()
        else:
            st.info("💡 Enhanced cost analysis not available.")
            show_basic_cost_summary()

    with tab4:
        show_risk_assessment_tab()

    with tab5:
        show_environment_analysis_tab()

    with tab6:
        show_visualizations_tab()

    with tab7:
        show_ai_insights_tab()

    with tab8:
        show_timeline_analysis_tab()
    
    with tab9:
        show_enhanced_recommendations_dashboard()

def show_basic_cost_summary():
    """Show basic cost summary from analysis results"""
    
    if not st.session_state.analysis_results:
        st.error("No analysis results available.")
        return
    
    results = st.session_state.analysis_results
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        monthly_cost = results.get('monthly_aws_cost', 0)
        st.metric("Monthly AWS Cost", f"${monthly_cost:,.0f}")
    
    with col2:
        annual_cost = monthly_cost * 12
        st.metric("Annual AWS Cost", f"${annual_cost:,.0f}")
    
    with col3:
        migration_cost = results.get('migration_costs', {}).get('total', 0)
        st.metric("Migration Cost", f"${migration_cost:,.0f}")
    
    with col4:
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            growth_percent = st.session_state.growth_analysis['growth_summary']['total_3_year_growth_percent']
            st.metric("3-Year Growth", f"{growth_percent:.1f}%")
        else:
            total_cost = annual_cost + migration_cost
            st.metric("Total First Year", f"${total_cost:,.0f}")
    
    # Environment costs breakdown
    st.markdown("### 💰 Cost Breakdown by Environment")
    
    env_costs = results.get('environment_costs', {})
    if env_costs:
        for env_name, costs in env_costs.items():
            with st.expander(f"🏢 {env_name.title()} Environment"):
                
                # Check if it's enhanced results format
                if isinstance(costs, dict) and 'instance_cost' in costs:
                    # Enhanced format
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Instance Cost", f"${costs.get('instance_cost', 0):,.2f}/month")
                        st.metric("Storage Cost", f"${costs.get('storage_cost', 0):,.2f}/month")
                    
                    with col2:
                        st.metric("Reader Instances", f"${costs.get('reader_costs', 0):,.2f}/month")
                        st.metric("Backup Cost", f"${costs.get('backup_cost', 0):,.2f}/month")
                    
                    with col3:
                        total_env_cost = costs.get('total_monthly_cost', 
                                                 sum([costs.get(k, 0) for k in ['instance_cost', 'storage_cost', 'reader_costs', 'backup_cost']]))
                        st.metric("Total Monthly", f"${total_env_cost:,.2f}")
                        
                        
                       
                else:
                    # Simple format - just show the cost
                    if isinstance(costs, (int, float)):
                        st.metric("Monthly Cost", f"${costs:,.2f}")
                    else:
                        st.write("Cost information not available in expected format")
    
    # Migration timeline and costs
    if 'migration_costs' in results:
        st.markdown("### 🚀 Migration Investment")
        
        migration = results['migration_costs']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Schema Migration", f"${migration.get('schema_migration', 0):,.0f}")
        
        with col2:
            st.metric("Data Transfer", f"${migration.get('data_transfer', 0):,.0f}")
        
        with col3:
            st.metric("Testing & Validation", f"${migration.get('testing', 0):,.0f}")

def show_growth_analysis_dashboard():
    """Show comprehensive growth analysis dashboard"""
    
    st.markdown("### 📈 3-Year Growth Analysis & Projections")
    
    # Check if growth analysis exists
    if not hasattr(st.session_state, 'growth_analysis') or not st.session_state.growth_analysis:
        st.warning("⚠️ Growth analysis not available. Please run the analysis first.")
        
        # Show basic growth planning instead
        st.markdown("#### 🎯 Growth Planning Preview")
        st.info("""
        **Growth analysis will show:**
        - 3-year cost projections with growth factors
        - Resource scaling requirements
        - Seasonal peak planning
        - Cost optimization opportunities
        - Scaling recommendations by year
        
        Run the migration analysis to see detailed growth projections.
        """)
        return
    
    growth_analysis = st.session_state.growth_analysis
    growth_summary = growth_analysis['growth_summary']
    
    # Key Growth Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "3-Year Growth",
            f"{growth_summary['total_3_year_growth_percent']:.1f}%",
            delta=f"CAGR: {growth_summary['compound_annual_growth_rate']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Current Annual Cost",
            f"${growth_summary['year_0_cost']:,.0f}",
            delta="Baseline"
        )
    
    with col3:
        st.metric(
            "Year 3 Projected Cost",
            f"${growth_summary['year_3_cost']:,.0f}",
            delta=f"+${growth_summary['year_3_cost'] - growth_summary['year_0_cost']:,.0f}"
        )
    
    with col4:
        st.metric(
            "Total 3-Year Investment",
            f"${growth_summary['total_3_year_investment']:,.0f}",
            delta=f"Avg: ${growth_summary['average_annual_cost']:,.0f}/year"
        )
    
    # Growth Projection Charts
    st.markdown("#### 📊 Growth Projections")
    
    try:
        charts = create_growth_projection_charts(growth_analysis)
        for chart in charts:
            st.plotly_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating growth charts: {str(e)}")
        
        # Show basic growth table instead
        st.markdown("#### 📈 Year-by-Year Projections")
        projections = growth_analysis['yearly_projections']
        
        years_data = []
        for year in range(4):
            year_data = projections[f'year_{year}']
            years_data.append({
                'Year': f'Year {year}' if year > 0 else 'Current',
                'Monthly Cost': f"${year_data['total_monthly']:,.0f}",
                'Annual Cost': f"${year_data['total_annual']:,.0f}",
                'Peak Cost': f"${year_data['peak_annual']:,.0f}"
            })
        
        st.table(years_data)
    
    # Scaling Recommendations
    st.markdown("#### 🎯 Scaling Recommendations")
    
    recommendations = growth_analysis.get('scaling_recommendations', [])
    
    if recommendations:
        for rec in recommendations:
            priority_color = {
                'High': '#e53e3e',
                'Medium': '#d69e2e',
                'Low': '#38a169'
            }.get(rec['priority'], '#666666')
            
            st.markdown(f"""
            <div style="border-left: 4px solid {priority_color}; padding: 15px; margin: 10px 0; background: {priority_color}22;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: {priority_color};">{rec['type']} (Year {rec['year']})</strong><br>
                        {rec['description']}<br>
                        <em>Action: {rec['action']}</em>
                    </div>
                    <div style="text-align: right;">
                        <strong>Priority: {rec['priority']}</strong><br>
                        <span style="color: #38a169;">Potential Savings: ${rec['estimated_savings']:,.0f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical scaling issues identified in the 3-year projection.")
        
        # Show default recommendations
        st.markdown("#### 💡 General Growth Recommendations")
        default_recommendations = [
            "📊 **Monitor growth trends** - Track actual vs projected growth quarterly",
            "💰 **Reserved Instances** - Consider 1-3 year commitments for 30-40% savings",
            "🔄 **Auto-scaling** - Implement auto-scaling for variable workloads",
            "📈 **Capacity planning** - Review and adjust capacity every 6 months",
            "🗄️ **Data lifecycle** - Implement archiving for older data to reduce storage costs"
        ]
        
        for rec in default_recommendations:
            st.markdown(rec)

def show_risk_assessment_tab():
    """Show risk assessment results"""
    
    if hasattr(st.session_state, 'risk_assessment') and st.session_state.risk_assessment:
        show_risk_assessment_robust()
    else:
        st.warning("⚠️ Risk assessment not available. Please run the migration analysis first.")

def show_environment_analysis_tab():
    """Show environment analysis"""
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Environment analysis not available. Please run the migration analysis first.")
        return
        
    st.markdown("### 🏢 Environment Analysis")
    
    # Show environment specifications
    if hasattr(st.session_state, 'environment_specs') and st.session_state.environment_specs:
        st.markdown("#### 📋 Environment Specifications")
        
        for env_name, specs in st.session_state.environment_specs.items():
            with st.expander(f"🏢 {env_name.title()} Environment"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Specs:**")
                    st.write(f"CPU Cores: {specs.get('cpu_cores', 'N/A')}")
                    st.write(f"RAM: {specs.get('ram_gb', 'N/A')} GB")
                    st.write(f"Storage: {specs.get('storage_gb', 'N/A')} GB")
                    st.write(f"Daily Usage: {specs.get('daily_usage_hours', 'N/A')} hours")
                
                with col2:
                    st.markdown("**Workload:**")
                    st.write(f"Peak Connections: {specs.get('peak_connections', 'N/A')}")
                    st.write(f"Workload Pattern: {specs.get('workload_pattern', 'N/A')}")
                    st.write(f"Environment Type: {specs.get('environment_type', 'N/A')}")
    
    # Show recommendations if available
    if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
        st.markdown("#### 💡 Environment Recommendations")
        
        recommendations = st.session_state.recommendations
        for env_name, rec in recommendations.items():
            with st.expander(f"🎯 {env_name.title()} Recommendations"):
                if isinstance(rec, dict):
                    st.markdown(f"**Recommended Instance:** {rec.get('instance_class', 'N/A')}")
                    st.markdown(f"**Environment Type:** {rec.get('environment_type', 'N/A')}")
                    st.markdown(f"**Multi-AZ:** {'Yes' if rec.get('multi_az', False) else 'No'}")
                    
                    if 'reasoning' in rec:
                        st.markdown("**Reasoning:**")
                        st.write(rec['reasoning'])

def show_visualizations_tab():
    """Show visualization charts"""
    
    st.markdown("### 📊 Cost & Performance Visualizations")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Visualizations not available. Please run the migration analysis first.")
        return
    
    try:
        # Cost comparison chart
        results = st.session_state.analysis_results
        
        # Create simple cost visualization
        env_costs = results.get('environment_costs', {})
        
        if env_costs:
            # Environment cost comparison
            env_names = []
            monthly_costs = []
            
            for env_name, costs in env_costs.items():
                env_names.append(env_name.title())
                
                if isinstance(costs, dict):
                    cost = costs.get('total_monthly_cost', 
                                   sum([costs.get(k, 0) for k in ['instance_cost', 'storage_cost', 'reader_costs', 'backup_cost']]))
                else:
                    cost = float(costs) if costs else 0
                
                monthly_costs.append(cost)
            
            if env_names and monthly_costs:
                fig = go.Figure(data=[
                    go.Bar(x=env_names, y=monthly_costs, marker_color='#3182ce')
                ])
                
                fig.update_layout(
                    title='Monthly Cost by Environment',
                    xaxis_title='Environment',
                    yaxis_title='Monthly Cost ($)',
                    height=400
                )
                
                # FIXED: Added unique key
                st.plotly_chart(fig, use_container_width=True, key="env_cost_visualization_chart")
        
        # Growth visualization if available
        if hasattr(st.session_state, 'growth_analysis') and st.session_state.growth_analysis:
            st.markdown("#### 📈 Growth Projections")
            try:
                charts = create_growth_projection_charts(st.session_state.growth_analysis)
                # FIXED: Added unique keys for each chart
                for i, chart in enumerate(charts):
                    st.plotly_chart(chart, use_container_width=True, key=f"viz_growth_chart_{i}")
            except Exception as e:
                st.error(f"Error creating growth charts: {str(e)}")
        
           # Enhanced cost chart if available
        if hasattr(st.session_state, 'enhanced_cost_chart') and st.session_state.enhanced_cost_chart:
            st.markdown("#### 💎 Enhanced Cost Analysis")
            # FIXED: Added unique key
            st.plotly_chart(st.session_state.enhanced_cost_chart, use_container_width=True, key="enhanced_cost_visualization_chart")
        
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")

def display_optimization_results(optimization_results: Dict):
    """Display comprehensive optimization results in Streamlit"""
    
    st.markdown("# 🚀 Optimized Reader/Writer Recommendations")
    
    # Overall summary
    total_monthly_cost = sum([env['cost_analysis']['monthly_breakdown']['total'] 
                             for env in optimization_results.values()])
    total_annual_cost = total_monthly_cost * 12
    avg_optimization_score = sum([env['optimization_score'] 
                                 for env in optimization_results.values()]) / len(optimization_results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Monthly Cost", f"${total_monthly_cost:,.0f}")
    
    with col2:
        st.metric("Total Annual Cost", f"${total_annual_cost:,.0f}")
    
    with col3:
        st.metric("Avg Optimization Score", f"{avg_optimization_score:.1f}/100")
    
    with col4:
        total_instances = sum([1 + env['reader_optimization']['count'] 
                              for env in optimization_results.values()])
        st.metric("Total Instances", total_instances)
    
    # Detailed results for each environment
    for env_name, optimization in optimization_results.items():
        with st.expander(f"🏢 {env_name.title()} - Optimization Score: {optimization['optimization_score']:.1f}/100", expanded=True):
            
            # Configuration overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✍️ Writer Configuration")
                writer = optimization['writer_optimization']
                st.info(f"""
                **Instance:** {writer['instance_class']}  
                **vCPUs:** {writer['specs'].vcpu}  
                **Memory:** {writer['specs'].memory_gb} GB  
                **Network:** {writer['specs'].network_performance}  
                **Multi-AZ:** {'✅ Yes' if writer['multi_az'] else '❌ No'}  
                **Monthly Cost:** ${writer['monthly_cost']:,.0f}  
                **Annual Cost:** ${writer['annual_cost']:,.0f}
                """)
                
                st.markdown("**Reasoning:**")
                st.write(writer['reasoning'])
            
            with col2:
                st.markdown("#### 📖 Reader Configuration")
                reader = optimization['reader_optimization']
                
                if reader['count'] > 0:
                    st.success(f"""
                    **Count:** {reader['count']} replicas  
                    **Instance:** {reader['instance_class']}  
                    **vCPUs:** {reader['specs'].vcpu} each  
                    **Memory:** {reader['specs'].memory_gb} GB each  
                    **Multi-AZ:** {'✅ Yes' if reader['multi_az'] else '❌ No'}  
                    **Cost per Reader:** ${reader['single_reader_monthly_cost']:,.0f}/month  
                    **Total Monthly Cost:** ${reader['total_monthly_cost']:,.0f}  
                    **Total Annual Cost:** ${reader['total_annual_cost']:,.0f}
                    """)
                else:
                    st.warning("**No read replicas recommended**")
                
                st.markdown("**Reasoning:**")
                st.write(reader['reasoning'])
            
            # Cost breakdown chart
            st.markdown("#### 💰 Cost Breakdown")
            
            cost_data = optimization['cost_analysis']['monthly_breakdown']
            
            fig = go.Figure(data=[go.Pie(
                labels=['Writer Instance', 'Reader Instances', 'Storage', 'Backup', 'Monitoring', 'Cross-AZ Transfer'],
                values=[cost_data['writer_instance'], cost_data['reader_instances'], 
                       cost_data['storage'], cost_data['backup'], cost_data['monitoring'], cost_data['cross_az_transfer']],
                hole=0.4,
                textinfo='label+percent+value',
                texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}'
            )])
            
            fig.update_layout(
                title=f"{env_name} Monthly Cost Distribution - Total: ${cost_data['total']:,.0f}",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"optimization_cost_chart_{env_name}")
            
            # Reserved Instance savings
            st.markdown("#### 💳 Reserved Instance Savings Opportunities")
            
            ri_1_year = optimization['cost_analysis']['reserved_instance_options']['1_year']
            ri_3_year = optimization['cost_analysis']['reserved_instance_options']['3_year']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1-Year Reserved Instances**")
                st.metric("Annual Savings", f"${ri_1_year['total_savings']:,.0f}", 
                         delta=f"{ri_1_year['discount_percentage']:.0f}% discount")
                st.write(f"Monthly cost: ${ri_1_year['monthly_cost']:,.0f}")
            
            with col2:
                st.markdown("**3-Year Reserved Instances**")
                st.metric("Total 3-Year Savings", f"${ri_3_year['total_savings']:,.0f}", 
                         delta=f"{ri_3_year['discount_percentage']:.0f}% discount")
                st.write(f"Monthly cost: ${ri_3_year['monthly_cost']:,.0f}")


def show_ai_insights_tab():
    """Show AI insights if available"""
    
    if hasattr(st.session_state, 'ai_insights') and st.session_state.ai_insights:
        st.markdown("### 🤖 AI-Powered Insights")
        
        insights = st.session_state.ai_insights
        
        if 'error' in insights:
            st.warning(f"AI insights partially available: {insights.get('summary', 'Analysis complete')}")
            st.error(f"Error: {insights['error']}")
        else:
            # Show AI insights
            if 'summary' in insights:
                st.markdown("#### 📝 Executive Summary")
                st.write(insights['summary'])
            
            if 'recommendations' in insights:
                st.markdown("#### 💡 AI Recommendations")
                for rec in insights['recommendations']:
                    st.markdown(f"• {rec}")
            
            if 'cost_optimization' in insights:
                st.markdown("#### 💰 Cost Optimization")
                st.write(insights['cost_optimization'])
    else:
        st.info("🤖 AI insights not available. Provide an Anthropic API key in the configuration to enable AI-powered analysis.")

def show_timeline_analysis_tab():
    """Show migration timeline analysis"""
    
    st.markdown("### 📅 Migration Timeline & Milestones")
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Timeline analysis not available. Please configure migration parameters first.")
        return
    
    params = st.session_state.migration_params
    timeline_weeks = params.get('migration_timeline_weeks', 12)
    
    # Create timeline phases
    phases = [
        {"phase": "Planning & Assessment", "weeks": max(2, timeline_weeks * 0.15), "description": "Requirements gathering, current state analysis"},
        {"phase": "Schema Migration", "weeks": max(2, timeline_weeks * 0.25), "description": "Convert schema, stored procedures, functions"},
        {"phase": "Data Migration", "weeks": max(3, timeline_weeks * 0.35), "description": "Initial load, incremental sync, validation"},
        {"phase": "Testing & Validation", "weeks": max(2, timeline_weeks * 0.15), "description": "Performance testing, data validation, UAT"},
        {"phase": "Go-Live & Support", "weeks": max(1, timeline_weeks * 0.10), "description": "Cutover, monitoring, post-migration support"}
    ]
    
    # Timeline visualization
    current_week = 0
    for i, phase in enumerate(phases):
        start_week = current_week
        end_week = current_week + phase["weeks"]
        
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            st.markdown(f"**{phase['phase']}**")
        
        with col2:
            st.markdown(f"Weeks {int(start_week)+1}-{int(end_week)}")
        
        with col3:
            st.markdown(phase['description'])
        
        current_week = end_week
    
    # Key milestones
    st.markdown("#### 🎯 Key Milestones")
    
    milestones = [
        f"Week {int(phases[0]['weeks'])}: Assessment Complete",
        f"Week {int(phases[0]['weeks'] + phases[1]['weeks'])}: Schema Migration Complete",
        f"Week {int(sum(p['weeks'] for p in phases[:3]))}: Data Migration Complete",
        f"Week {int(sum(p['weeks'] for p in phases[:4]))}: Testing Complete",
        f"Week {timeline_weeks}: Go-Live"
    ]
    
    for milestone in milestones:
        st.markdown(f"• {milestone}")
    
    # Team and resources
    st.markdown("#### 👥 Team & Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Team Configuration:**")
        st.write(f"Team Size: {params.get('team_size', 5)} people")
        st.write(f"Expertise Level: {params.get('team_expertise', 'medium').title()}")
        st.write(f"Timeline: {timeline_weeks} weeks")
    
    with col2:
        st.markdown("**Migration Budget:**")
        budget = params.get('migration_budget', 500000)
        st.write(f"Total Budget: ${budget:,.0f}")
        weekly_budget = budget / timeline_weeks if timeline_weeks > 0 else 0
        st.write(f"Weekly Budget: ${weekly_budget:,.0f}")

def show_reports_section():
    """Show reports and export section - ROBUST VERSION"""
    
    st.markdown("## 📄 Reports & Export")
    
    # Check for both regular and enhanced analysis results
    has_regular_results = st.session_state.analysis_results is not None
    has_enhanced_results = hasattr(st.session_state, 'enhanced_analysis_results') and st.session_state.enhanced_analysis_results is not None
    
    if not has_regular_results and not has_enhanced_results:
        st.warning("⚠️ Please complete the analysis first to generate reports.")
        st.info("👆 Go to 'Analysis & Recommendations' section and click '🚀 Run Comprehensive Analysis'")
        
        # Show current status
        st.markdown("### 📊 Current Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            config_status = "✅ Complete" if st.session_state.migration_params else "❌ Missing"
            st.metric("Migration Config", config_status)
        
        with col2:
            env_status = "✅ Complete" if st.session_state.environment_specs else "❌ Missing"
            st.metric("Environment Setup", env_status)
        
        with col3:
            analysis_status = "✅ Complete" if (has_regular_results or has_enhanced_results) else "❌ Pending"
            st.metric("Analysis Results", analysis_status)
        
        return
    
    # Determine which results to use
    if has_enhanced_results:
        results = st.session_state.enhanced_analysis_results
        recommendations = getattr(st.session_state, 'enhanced_recommendations', {})
        st.info("📊 Using Enhanced Analysis Results")
    else:
        results = st.session_state.analysis_results
        recommendations = getattr(st.session_state, 'recommendations', {})
        st.info("📊 Using Standard Analysis Results")
    
    # Report generation options
    st.markdown("### 📊 Available Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👔 Executive Summary")
        st.markdown("Perfect for stakeholders and decision makers")
        st.markdown("**Includes:**")
        st.markdown("• High-level cost analysis")
        st.markdown("• ROI and timeline overview")
        st.markdown("• Risk summary")
        st.markdown("• Key recommendations")
        
        if st.button("📄 Generate Executive PDF", key="exec_pdf", use_container_width=True):
            with st.spinner("Generating executive summary..."):
                try:
                    pdf_buffer = generate_executive_summary_pdf_robust(
                        results,
                        st.session_state.migration_params
                    )
                    
                    if pdf_buffer:
                        st.download_button(
                            label="📥 Download Executive Summary",
                            data=pdf_buffer.getvalue(),
                            file_name=f"AWS_Migration_Executive_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to generate PDF")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
    
    with col2:
        st.markdown("#### 🔧 Technical Report")
        st.markdown("Detailed technical analysis for architects and engineers")
        st.markdown("**Includes:**")
        st.markdown("• Environment specifications")
        st.markdown("• Instance recommendations")
        st.markdown("• Detailed cost breakdown")
        st.markdown("• Technical considerations")
        
        if st.button("📄 Generate Technical PDF", key="tech_pdf", use_container_width=True):
            with st.spinner("Generating technical report..."):
                try:
                    pdf_buffer = generate_technical_report_pdf_robust(
                        results,
                        recommendations,
                        st.session_state.migration_params
                    )
                    
                    if pdf_buffer:
                        st.download_button(
                            label="📥 Download Technical Report",
                            data=pdf_buffer.getvalue(),
                            file_name=f"AWS_Migration_Technical_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to generate PDF")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
    
    with col3:
        st.markdown("#### 📊 Data Export")
        st.markdown("Raw data for further analysis")
        st.markdown("**Includes:**")
        st.markdown("• Cost analysis data")
        st.markdown("• Environment specifications")
        st.markdown("• Risk assessment data")
        st.markdown("• Recommendations")
        
        if st.button("📊 Export Data (CSV)", key="csv_export", use_container_width=True):
            try:
                # Prepare CSV data
                csv_data = prepare_csv_export_data(results, recommendations)
                
                if csv_data:
                    csv_string = csv_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CSV Data",
                        data=csv_string,
                        file_name=f"AWS_Migration_Analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("No data available for export")
            except Exception as e:
                st.error(f"Error preparing CSV: {str(e)}")
    
    # Bulk download option
    st.markdown("---")
    st.markdown("### 📦 Bulk Download")
    
    if st.button("📊 Generate All Reports", key="bulk_reports", use_container_width=True):
        with st.spinner("Generating all reports... This may take a moment..."):
            try:
                # Create ZIP file with all reports
                zip_buffer = create_bulk_reports_zip(results, recommendations, st.session_state.migration_params)
                
                if zip_buffer:
                    st.download_button(
                        label="📥 Download All Reports (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"AWS_Migration_Complete_Analysis_{datetime.now().strftime('%Y%m%d')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                else:
                    st.error("Failed to generate reports package")
            except Exception as e:
                st.error(f"Error generating bulk reports: {str(e)}")

# ===========================
# COST RECONCILIATION INTERFACE
# ===========================

def reconcile_costs_with_aws_analysis():
    """Main reconciliation function"""
    
    st.markdown("### 🔄 Cost Reconciliation with AWS Cost Analysis")
    
    # Input AWS Cost Analysis data
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 AWS Cost Analysis Data")
        st.info("Enter your actual AWS Cost Analysis data below:")
        
        aws_total_cost = st.number_input(
            "Total Monthly Cost from AWS Cost Analysis ($)", 
            value=0.0, 
            min_value=0.0,
            help="Enter the total monthly cost shown in AWS Cost Analysis"
        )
        
        aws_instance_cost = st.number_input(
            "RDS Instance Costs ($)", 
            value=0.0, 
            min_value=0.0,
            help="RDS instance costs from Cost Analysis"
        )
        
        aws_storage_cost = st.number_input(
            "Storage Costs ($)", 
            value=0.0, 
            min_value=0.0,
            help="Storage and backup costs"
        )
        
        aws_data_transfer = st.number_input(
            "Data Transfer Costs ($)", 
            value=0.0, 
            min_value=0.0,
            help="Cross-AZ and internet data transfer"
        )
    
    with col2:
        st.markdown("#### 🤖 AI Predicted Costs")
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            ai_total_cost = results.get('monthly_aws_cost', 0)
            st.metric("AI Total Cost", f"${ai_total_cost:,.2f}")
            
            # Calculate variance
            if aws_total_cost > 0:
                variance = abs(aws_total_cost - ai_total_cost)
                variance_pct = (variance / aws_total_cost) * 100
                
                st.metric("Cost Variance", f"${variance:,.2f}", f"{variance_pct:.1f}%")
                
                # Show color-coded status
                if variance_pct <= 5:
                    st.success("✅ Excellent alignment (≤5% variance)")
                elif variance_pct <= 15:
                    st.warning("⚠️ Acceptable variance (5-15%)")
                else:
                    st.error("❌ High variance (>15%) - needs investigation")
            else:
                st.info("Enter AWS Cost Analysis data to compare")
        else:
            st.warning("Run migration analysis first to get AI predictions")
    
    # Reconciliation analysis
    if st.button("🔍 Analyze Cost Differences", type="primary"):
        if aws_total_cost > 0 and st.session_state.analysis_results:
            perform_cost_reconciliation(aws_total_cost, ai_total_cost, aws_instance_cost, aws_storage_cost, aws_data_transfer)
        else:
            st.error("Please enter AWS Cost Analysis data and run migration analysis first")

def perform_cost_reconciliation(aws_cost, ai_cost, aws_instance, aws_storage, aws_transfer):
    """Perform detailed cost reconciliation analysis"""
    
    st.markdown("### 📊 Detailed Cost Reconciliation Analysis")
    
    variance_threshold = 0.05  # 5% acceptable variance
    variance_pct = abs(aws_cost - ai_cost) / max(aws_cost, 1)
    
    # Overall assessment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AWS Cost Analysis", f"${aws_cost:,.2f}")
    
    with col2:
        st.metric("AI Prediction", f"${ai_cost:,.2f}")
    
    with col3:
        variance_dollars = abs(aws_cost - ai_cost)
        st.metric("Variance", f"${variance_dollars:,.2f}", f"{variance_pct*100:.1f}%")
    
    # Detailed breakdown comparison
    st.markdown("#### 🔍 Component-Level Analysis")
    
    results = st.session_state.analysis_results
    env_costs = results.get('environment_costs', {})
    
    # Calculate AI component costs
    ai_instance_total = sum([env.get('instance_cost', 0) for env in env_costs.values()])
    ai_storage_total = sum([env.get('storage_cost', 0) for env in env_costs.values()])
    ai_transfer_total = sum([env.get('transfer_cost', 0) for env in env_costs.values()])
    
    # Component comparison table
    comparison_data = [
        ['Cost Component', 'AWS Cost Analysis', 'AI Prediction', 'Variance', 'Status'],
        ['Instance Costs', f'${aws_instance:,.2f}', f'${ai_instance_total:,.2f}', 
         f'${abs(aws_instance - ai_instance_total):,.2f}', 
         get_variance_status(aws_instance, ai_instance_total)],
        ['Storage Costs', f'${aws_storage:,.2f}', f'${ai_storage_total:,.2f}', 
         f'${abs(aws_storage - ai_storage_total):,.2f}', 
         get_variance_status(aws_storage, ai_storage_total)],
        ['Data Transfer', f'${aws_transfer:,.2f}', f'${ai_transfer_total:,.2f}', 
         f'${abs(aws_transfer - ai_transfer_total):,.2f}', 
         get_variance_status(aws_transfer, ai_transfer_total)]
    ]
    
    # Display comparison table
    for i, row in enumerate(comparison_data):
        if i == 0:  # Header
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"**{row[0]}**")
            with col2:
                st.markdown(f"**{row[1]}**")
            with col3:
                st.markdown(f"**{row[2]}**")
            with col4:
                st.markdown(f"**{row[3]}**")
            with col5:
                st.markdown(f"**{row[4]}**")
        else:  # Data rows
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.write(row[0])
            with col2:
                st.write(row[1])
            with col3:
                st.write(row[2])
            with col4:
                st.write(row[3])
            with col5:
                if row[4] == '✅ Good':
                    st.success(row[4])
                elif row[4] == '⚠️ Check':
                    st.warning(row[4])
                else:
                    st.error(row[4])
    
    # Reconciliation recommendations
    st.markdown("#### 🔧 Reconciliation Recommendations")
    
    if variance_pct <= variance_threshold:
        st.success(f"✅ Costs are aligned within {variance_threshold*100}% tolerance")
        st.markdown("**Your AI cost predictions are accurate!**")
    else:
        st.warning(f"⚠️ Cost variance of {variance_pct*100:.1f}% exceeds {variance_threshold*100}% threshold")
        
        # Provide specific reconciliation suggestions
        if ai_cost > aws_cost:
            st.markdown("**AI costs are higher than AWS Cost Analysis - Possible causes:**")
            st.markdown("• ✅ Check if AI includes Reserved Instance discounts")
            st.markdown("• ✅ Verify Multi-AZ calculations (should be 2x base cost)")
            st.markdown("• ✅ Review IOPS cost calculations for gp3/io2 storage")
            st.markdown("• ✅ Check for included support/monitoring costs")
            st.markdown("• ✅ Verify data transfer cost calculations")
        else:
            st.markdown("**AWS Cost Analysis is higher than AI predictions - Possible causes:**")
            st.markdown("• ✅ Verify if taxes and fees are included in AWS costs")
            st.markdown("• ✅ Check for additional AWS services not modeled by AI")
            st.markdown("• ✅ Review data transfer and cross-AZ costs")
            st.markdown("• ✅ Validate backup and snapshot costs")
            st.markdown("• ✅ Check for hidden costs (CloudWatch, Enhanced Monitoring)")
        
        # Action items
        st.markdown("#### 📋 Recommended Actions")
        action_items = [
            "Review AWS Cost Analysis filters and time period",
            "Verify all cost components are included in both calculations",
            "Check for region consistency between calculations",
            "Validate Multi-AZ and Reserved Instance settings",
            "Consider running analysis with reconciliation-aware pricing API"
        ]
        
        for action in action_items:
            st.markdown(f"• {action}")

def show_cost_reconciliation_page():
    """Show cost reconciliation page"""
    
    st.markdown("## 🔄 Cost Reconciliation & Validation")
    
    # Check prerequisites
    if not st.session_state.analysis_results:
        st.warning("⚠️ Please run the migration analysis first.")
        st.info("👆 Go to 'Analysis & Recommendations' section and run the comprehensive analysis")
        return
    
    # Create tabs for different reconciliation views
    tab1, tab2, tab3 = st.tabs([
        "🔍 AWS Cost Analysis Comparison",
        "📊 Cost Validation Dashboard", 
        "⚙️ Reconciliation Settings"
    ])
    
    with tab1:
        reconcile_costs_with_aws_analysis()
    
    with tab2:
        show_cost_validation_dashboard()
    
    with tab3:
        show_reconciliation_settings()

def show_reconciliation_settings():
    """Show reconciliation configuration settings"""
    
    st.markdown("### ⚙️ Reconciliation Settings")
    
    st.markdown("Configure how cost calculations align with AWS Cost Analysis:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 Cost Calculation Settings")
        
        include_taxes = st.checkbox(
            "Include Taxes in Calculations", 
            value=True,
            help="Include estimated taxes to match AWS billing"
        )
        
        include_support = st.checkbox(
            "Include Support Costs", 
            value=False,
            help="Include AWS support plan costs (typically 10% of infrastructure)"
        )
        
        enterprise_discount = st.checkbox(
            "Apply Enterprise Discount", 
            value=False,
            help="Apply enterprise discount rate (typically 5-10%)"
        )
        
        if enterprise_discount:
            discount_rate = st.slider(
                "Enterprise Discount Rate (%)",
                min_value=1, max_value=20, value=5
            )
    
    with col2:
        st.markdown("#### 🔧 Reconciliation Preferences")
        
        variance_threshold = st.slider(
            "Acceptable Variance Threshold (%)",
            min_value=1, max_value=25, value=5,
            help="Variance threshold for cost alignment"
        )
        
        reconciliation_method = st.selectbox(
            "Reconciliation Method",
            ["Real-time AWS API", "Manual Input", "Hybrid"],
            help="How to obtain AWS cost data for comparison"
        )
        
        auto_reconcile = st.checkbox(
            "Auto-reconcile on Analysis",
            value=False,
            help="Automatically attempt reconciliation when running analysis"
        )
    
    # Save settings
    if st.button("💾 Save Reconciliation Settings", type="primary"):
        reconciliation_settings = {
            'include_taxes': include_taxes,
            'include_support': include_support,
            'enterprise_discount': enterprise_discount,
            'discount_rate': discount_rate if enterprise_discount else 0,
            'variance_threshold': variance_threshold,
            'reconciliation_method': reconciliation_method,
            'auto_reconcile': auto_reconcile
        }
        
        st.session_state.reconciliation_settings = reconciliation_settings
        st.success("✅ Reconciliation settings saved!")


def get_variance_status(aws_value, ai_value):
    """Get status indicator for cost variance"""
    if aws_value == 0 and ai_value == 0:
        return '➖ N/A'
    
    variance_pct = abs(aws_value - ai_value) / max(aws_value, ai_value, 1) * 100
    
    if variance_pct <= 5:
        return '✅ Good'
    elif variance_pct <= 15:
        return '⚠️ Check'
    else:
        return '❌ High'

def show_cost_validation_dashboard():
    """Dashboard to validate cost predictions against actual AWS billing"""
    
    st.markdown("### 📊 Cost Validation Dashboard")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Run migration analysis first to enable cost validation")
        return
    
    st.info("💡 Use this dashboard to compare AI predictions with your actual AWS billing data")
    
    # Create comparison table
    comparison_data = []
    
    env_costs = st.session_state.analysis_results.get('environment_costs', {})
    
    for env_name, costs in env_costs.items():
        ai_predicted = costs.get('total_monthly', 0)
        comparison_data.append({
            'Environment': env_name,
            'AI Predicted ($)': ai_predicted,
            'AWS Actual ($)': 0.0,  # User will enter this
            'Variance ($)': 0.0,
            'Variance (%)': 0.0,
            'Status': 'Pending'
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Editable dataframe for AWS actual costs
    st.markdown("#### ✏️ Enter Your Actual AWS Costs")
    
    edited_df = st.data_editor(
        df,
        column_config={
            "AWS Actual ($)": st.column_config.NumberColumn(
                "AWS Actual ($)",
                help="Enter actual costs from AWS billing/Cost Analysis",
                min_value=0.0,
                format="%.2f"
            )
        },
        disabled=["Environment", "AI Predicted ($)", "Variance ($)", "Variance (%)", "Status"],
        use_container_width=True,
        key="cost_validation_editor"
    )
    
    # Calculate variances
    if st.button("📊 Calculate Variances", type="primary"):
        calculate_and_display_variances(edited_df)

def calculate_and_display_variances(df):
    """Calculate and display cost variances"""
    
    st.markdown("#### 📈 Variance Analysis Results")
    
    # Calculate variances for each row
    results_data = []
    total_ai_cost = 0
    total_aws_cost = 0
    
    for _, row in df.iterrows():
        ai_cost = row['AI Predicted ($)']
        aws_cost = row['AWS Actual ($)']
        
        total_ai_cost += ai_cost
        total_aws_cost += aws_cost
        
        if aws_cost > 0:
            variance_dollars = abs(ai_cost - aws_cost)
            variance_pct = (variance_dollars / aws_cost) * 100
            
            # Determine status
            if variance_pct <= 5:
                status = '✅ Excellent'
                status_color = 'success'
            elif variance_pct <= 15:
                status = '⚠️ Acceptable'
                status_color = 'warning'
            else:
                status = '❌ High Variance'
                status_color = 'error'
            
            results_data.append({
                'Environment': row['Environment'],
                'AI Predicted': f"${ai_cost:,.2f}",
                'AWS Actual': f"${aws_cost:,.2f}",
                'Variance': f"${variance_dollars:,.2f}",
                'Variance %': f"{variance_pct:.1f}%",
                'Status': status,
                'Color': status_color
            })
    
    # Display results
    for result in results_data:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.write(result['Environment'])
        with col2:
            st.write(result['AI Predicted'])
        with col3:
            st.write(result['AWS Actual'])
        with col4:
            st.write(result['Variance'])
        with col5:
            st.write(result['Variance %'])
        with col6:
            if result['Color'] == 'success':
                st.success(result['Status'])
            elif result['Color'] == 'warning':
                st.warning(result['Status'])
            else:
                st.error(result['Status'])
    
    # Overall summary
    if total_aws_cost > 0:
        overall_variance = abs(total_ai_cost - total_aws_cost)
        overall_variance_pct = (overall_variance / total_aws_cost) * 100
        
        st.markdown("#### 🎯 Overall Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total AI Predicted", f"${total_ai_cost:,.2f}")
        with col2:
            st.metric("Total AWS Actual", f"${total_aws_cost:,.2f}")
        with col3:
            st.metric("Total Variance", f"${overall_variance:,.2f}")
        with col4:
            st.metric("Overall Variance %", f"{overall_variance_pct:.1f}%")
        
        # Recommendations
        if overall_variance_pct <= 5:
            st.success("🎉 Excellent! Your AI cost predictions are highly accurate.")
        elif overall_variance_pct <= 15:
            st.warning("👍 Good accuracy. Minor adjustments may improve predictions.")
        else:
            st.error("⚠️ High variance detected. Consider reviewing cost calculation methodology.")


def show_optimized_recommendations():
    """Show optimized Reader/Writer recommendations"""
    
    st.markdown("## 🧠 AI-Optimized Reader/Writer Recommendations")
    
    if not st.session_state.environment_specs:
        st.warning("⚠️ Please configure environments first.")
        st.info("👆 Go to 'Environment Setup' section to configure your database environments")
        return
    
    # Show current environment count
    env_count = len(st.session_state.environment_specs)
    is_enhanced = is_enhanced_environment_data(st.session_state.environment_specs)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Environments Configured", env_count)
    with col2:
        config_type = "Enhanced Cluster Config" if is_enhanced else "Basic Config"
        st.metric("Configuration Type", config_type)
    
    if not is_enhanced:
        st.info("💡 For best results, use the Enhanced Environment Setup with cluster configuration")
    
    # Configuration options
    st.markdown("### ⚙️ Optimization Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        optimization_goal = st.selectbox(
            "Optimization Goal",
            ["Cost Efficiency", "Performance", "Balanced"],
            index=2,
            help="Choose whether to prioritize cost savings, performance, or a balance of both"
        )
    
    with col2:
        consider_reserved_instances = st.checkbox(
            "Include Reserved Instance Analysis",
            value=True,
            help="Calculate potential savings with 1-year and 3-year Reserved Instances"
        )
    
    with col3:
        environment_priority = st.selectbox(
            "Environment Priority",
            ["Production First", "All Equal", "Cost First"],
            index=0,
            help="Prioritization strategy for resource allocation"
        )
    
    # Run optimization button
    if st.button("🧠 Generate AI Recommendations", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing workloads and optimizing configurations..."):
            
            try:
                # Initialize optimizer
                optimizer = OptimizedReaderWriterAnalyzer()
                
                # Apply optimization settings
                optimizer.set_optimization_preferences(
                    goal=optimization_goal.lower().replace(" ", "_"),
                    consider_reserved=consider_reserved_instances,
                    environment_priority=environment_priority.lower().replace(" ", "_")
                )
                
                # Run optimization
                optimization_results = optimizer.optimize_cluster_configuration(st.session_state.environment_specs)
                
                # Store results
                st.session_state.optimization_results = optimization_results
                
                st.success("✅ Optimization complete!")
                
                # Show quick summary
                total_monthly = sum([env['cost_analysis']['monthly_breakdown']['total'] 
                                   for env in optimization_results.values()])
                avg_score = sum([env['optimization_score'] 
                               for env in optimization_results.values()]) / len(optimization_results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Monthly Cost", f"${total_monthly:,.0f}")
                with col2:
                    st.metric("Average Optimization Score", f"{avg_score:.1f}/100")
                with col3:
                    potential_ri_savings = sum([
                        env['cost_analysis']['reserved_instance_options']['3_year']['total_savings']
                        for env in optimization_results.values()
                    ])
                    st.metric("Potential 3-Year RI Savings", f"${potential_ri_savings:,.0f}")
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.code(str(e))
    
    # Display results if available
    if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
        st.markdown("---")
        display_optimization_results(st.session_state.optimization_results)
        
def show_vrops_main_page():
    """Main vROps integration page"""
    
    st.markdown("## 📡 vROps Integration")
    st.markdown("Extract server and performance data from your on-premise vROps installation")
    
    # Check if already connected
    if st.session_state.vrops_servers:
        show_vrops_dashboard()
    else:
        show_vrops_connection_form()

def show_vrops_connection_form():
    """Simple vROps connection form"""
    
    st.markdown("### 🔌 Connect to vROps")
    
    with st.form("vrops_connection"):
        col1, col2 = st.columns(2)
        
        with col1:
            vrops_host = st.text_input(
                "vROps Server",
                placeholder="vrops.company.com",
                help="IP address or hostname of your vROps server"
            )
            
            username = st.text_input(
                "Username", 
                placeholder="admin@vsphere.local"
            )
        
        with col2:
            password = st.text_input(
                "Password",
                type="password"
            )
            
            hours_back = st.selectbox(
                "Data Collection Period",
                [24, 48, 168],  # 1 day, 2 days, 1 week
                index=2,
                format_func=lambda x: f"{x} hours ({x//24} days)"
            )
        
        # Advanced options
        with st.expander("Advanced Options"):
            verify_ssl = st.checkbox(
                "Verify SSL Certificate", 
                value=False,
                help="Uncheck for self-signed certificates"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                collect_vms = st.checkbox("Collect Virtual Machines", value=True)
            with col2:
                auto_identify_db = st.checkbox("Auto-identify Database Servers", value=True)
        
        submitted = st.form_submit_button("🔌 Connect & Extract Data", use_container_width=True)
        
        if submitted:
            if not all([vrops_host, username, password]):
                st.error("Please fill in all required fields")
                return
            
            # Extract data
            with st.spinner("Connecting to vROps and extracting data..."):
                success = simple_vrops_extraction(
                    vrops_host, username, password, verify_ssl, 
                    hours_back, collect_vms, auto_identify_db
                )
                
                if success:
                    st.success("✅ Successfully connected and extracted data!")
                    st.experimental_rerun()
                else:
                    st.error("❌ Connection failed. Please check your credentials.")

# =============================================================================
# STEP 3: SIMPLIFIED vROPS DATA EXTRACTION
# =============================================================================

def simple_vrops_extraction(vrops_host, username, password, verify_ssl, 
                           hours_back, collect_vms, auto_identify_db):
    """Simplified vROps data extraction"""
    
    try:
        # Basic vROps API connection
        session = requests.Session()
        base_url = f"https://{vrops_host}/suite-api/api"
        
        # Authenticate
        auth_url = f"{base_url}/auth/token/acquire"
        auth_data = {"username": username, "password": password}
        
        response = session.post(auth_url, json=auth_data, verify=verify_ssl, timeout=30)
        
        if response.status_code != 200:
            st.error(f"Authentication failed: {response.status_code}")
            return False
        
        token = response.json().get('token')
        session.headers.update({
            'Authorization': f'vRealizeOpsToken {token}',
            'Content-Type': 'application/json'
        })
        
        st.success("✅ Authentication successful")
        
        # Get VMs
        vms_url = f"{base_url}/resources"
        params = {
            'adapterKind': 'VMWARE',
            'resourceKind': 'VirtualMachine',
            'pageSize': 100
        }
        
        response = session.get(vms_url, params=params, verify=verify_ssl, timeout=60)
        
        if response.status_code != 200:
            st.error(f"Failed to get VMs: {response.status_code}")
            return False
        
        vms = response.json().get('resourceList', [])
        st.info(f"Found {len(vms)} virtual machines")
        
        # Process VMs
        servers = []
        server_metrics = {}
        analysis_results = {}
        
        progress_bar = st.progress(0)
        
        for i, vm in enumerate(vms[:20]):  # Limit to first 20 VMs for demo
            progress_bar.progress((i + 1) / min(len(vms), 20))
            
            vm_id = vm.get('identifier')
            vm_name = vm.get('resourceKey', {}).get('name', 'Unknown')
            
            # Get basic properties
            props_url = f"{base_url}/resources/{vm_id}/properties"
            response = session.get(props_url, verify=verify_ssl, timeout=30)
            
            properties = {}
            if response.status_code == 200:
                for prop in response.json().get('property', []):
                    properties[prop['name']] = prop.get('value', '')
            
            # Create server object
            server = {
                'id': vm_id,
                'name': vm_name,
                'type': 'Virtual Machine',
                'cpu_cores': safe_int(properties.get('config|hardware|num_Cpu', 2)),
                'memory_gb': safe_float(properties.get('config|hardware|memoryMB', 4096)) / 1024,
                'storage_gb': safe_float(properties.get('config|hardware|disk_Space', 100000000000)) / (1024**3),
                'os_type': properties.get('config|guestFullName', 'Unknown'),
                'power_state': properties.get('summary|runtime|powerState', 'Unknown')
            }
            
            servers.append(server)
            
            # Get basic performance metrics
            stats_url = f"{base_url}/resources/{vm_id}/stats"
            metric_params = {
                'statKey': ['cpu|usage_average', 'mem|usage_average'],
                'begin': int((datetime.now() - timedelta(hours=hours_back)).timestamp() * 1000),
                'end': int(datetime.now().timestamp() * 1000)
            }
            
            response = session.get(stats_url, params=metric_params, verify=verify_ssl, timeout=30)
            
            if response.status_code == 200:
                stats = response.json()
                metrics = extract_simple_metrics(stats)
                server_metrics[vm_id] = metrics
                
                # Simple analysis
                analysis = simple_performance_analysis(server, metrics)
                analysis_results[vm_id] = analysis
        
        # Identify database servers
        database_servers = []
        if auto_identify_db:
            for server in servers:
                if is_potential_database_server(server):
                    database_servers.append(server)
        
        # Store in session state
        st.session_state.vrops_servers = servers
        st.session_state.vrops_database_servers = database_servers
        st.session_state.vrops_metrics = server_metrics
        st.session_state.vrops_analysis = analysis_results
        st.session_state.vrops_connection_info = {
            'host': vrops_host,
            'extraction_time': datetime.now().isoformat(),
            'hours_back': hours_back
        }
        
        st.success(f"✅ Extracted data for {len(servers)} servers ({len(database_servers)} potential database servers)")
        return True
        
    except Exception as e:
        st.error(f"Error during extraction: {str(e)}")
        return False

# =============================================================================
# STEP 4: HELPER FUNCTIONS
# =============================================================================

def safe_int(value, default=0):
    """Safely convert to int"""
    try:
        return int(float(value)) if value else default
    except:
        return default

def safe_float(value, default=0.0):
    """Safely convert to float"""
    try:
        return float(value) if value else default
    except:
        return default

def extract_simple_metrics(stats_response):
    """Extract simple metrics from vROps stats response"""
    metrics = {
        'cpu_usage_avg': 0,
        'memory_usage_avg': 0,
        'health_score': 50
    }
    
    try:
        for value in stats_response.get('values', []):
            stat_key = value['statKey']['key']
            stat_values = [float(v['data']) for v in value.get('data', []) if v.get('data') is not None]
            
            if stat_values:
                avg_value = sum(stat_values) / len(stat_values)
                
                if stat_key == 'cpu|usage_average':
                    metrics['cpu_usage_avg'] = avg_value
                elif stat_key == 'mem|usage_average':
                    metrics['memory_usage_avg'] = avg_value
    except:
        pass
    
    return metrics

def simple_performance_analysis(server, metrics):
    """Simple performance analysis"""
    
    cpu_usage = metrics.get('cpu_usage_avg', 0)
    memory_usage = metrics.get('memory_usage_avg', 0)
    
    # Simple health scoring
    cpu_score = 100 if cpu_usage < 70 else 80 if cpu_usage < 85 else 50
    memory_score = 100 if memory_usage < 80 else 80 if memory_usage < 90 else 50
    overall_score = (cpu_score + memory_score) / 2
    
    # Simple AWS instance recommendation
    if server['memory_gb'] / server['cpu_cores'] > 6:  # Memory-optimized
        if server['cpu_cores'] <= 2:
            instance_class = 'db.r5.large'
        elif server['cpu_cores'] <= 4:
            instance_class = 'db.r5.xlarge'
        else:
            instance_class = 'db.r5.2xlarge'
    else:  # General purpose
        if server['cpu_cores'] <= 2:
            instance_class = 'db.t3.large'
        elif server['cpu_cores'] <= 4:
            instance_class = 'db.m5.xlarge'
        else:
            instance_class = 'db.m5.2xlarge'
    
    return {
        'server_name': server['name'],
        'health_scores': {
            'cpu': cpu_score,
            'memory': memory_score,
            'overall': overall_score
        },
        'aws_recommendations': {
            'instance_class': instance_class,
            'fit_score': overall_score
        },
        'performance_summary': {
            'cpu_usage_avg': cpu_usage,
            'memory_usage_avg': memory_usage
        },
        'migration_readiness': 'Ready' if overall_score > 70 else 'Needs Attention'
    }

def is_potential_database_server(server):
    """Simple database server identification"""
    name_lower = server['name'].lower()
    
    # Check for database keywords in name
    db_keywords = ['sql', 'oracle', 'postgres', 'mysql', 'db', 'database']
    has_db_keyword = any(keyword in name_lower for keyword in db_keywords)
    
    # Check for sufficient resources (databases typically need more RAM)
    sufficient_resources = server['memory_gb'] >= 8
    
    return has_db_keyword or sufficient_resources

# STEP 6: UPDATE YOUR SESSION STATE (find initialize_session_state function and add this line)
# Add this line to your defaults dictionary:
# 'optimization_results': None,

# STEP 7: UPDATE YOUR MAIN NAVIGATION (find your main() function and update the page radio)
# Add "🧠 AI Optimizer" to your page options and add the routing case:
# elif page == "🧠 AI Optimizer":
#     show_optimized_recommendations()

def prepare_csv_export_data(results, recommendations):
    """Prepare CSV data for export"""
    try:
        env_costs = results.get('environment_costs', {})
        if not env_costs:
            return None
        
        csv_data = []
        for env_name, costs in env_costs.items():
            # Handle both enhanced and regular cost structures
            if isinstance(costs, dict):
                # Get cost components safely
                instance_cost = costs.get('instance_cost', costs.get('writer_instance_cost', 0))
                storage_cost = costs.get('storage_cost', 0)
                backup_cost = costs.get('backup_cost', 0)
                total_monthly = costs.get('total_monthly', 0)
                
                # Additional enhanced fields
                reader_costs = costs.get('reader_costs', 0)
                reader_count = costs.get('reader_count', 0)
                
                csv_data.append({
                    'Environment': env_name,
                    'Instance_Cost': instance_cost,
                    'Reader_Costs': reader_costs,
                    'Reader_Count': reader_count,
                    'Storage_Cost': storage_cost,
                    'Backup_Cost': backup_cost,
                    'Total_Monthly': total_monthly,
                    'Total_Annual': total_monthly * 12
                })
        
        if csv_data:
            return pd.DataFrame(csv_data)
        else:
            return None
            
    except Exception as e:
        print(f"Error preparing CSV data: {e}")
        return None

def generate_executive_summary_pdf_robust(results, migration_params):
    """Generate executive summary PDF - ROBUST VERSION"""
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2d3748'),
            alignment=1
        )
        
        story = []
        
        # Title and header
        story.append(Paragraph("AWS Database Migration Analysis", title_style))
        story.append(Paragraph("Executive Summary Report", styles['Heading2']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key metrics table
        migration_costs = results.get('migration_costs', {})
        
        metrics_data = [
            ['Metric', 'Value', 'Impact'],
            ['Monthly AWS Cost', f"${results.get('monthly_aws_cost', 0):,.0f}", 'Operational'],
            ['Annual AWS Cost', f"${results.get('annual_aws_cost', results.get('monthly_aws_cost', 0) * 12):,.0f}", 'Budget Planning'],
            ['Migration Investment', f"${migration_costs.get('total', 0):,.0f}", 'One-time'],
            ['ROI Timeline', '18-24 months', 'Financial'],
            ['Risk Level', 'Medium', 'Manageable']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Key Recommendations", styles['Heading2']))
        
        recommendations = [
            "Proceed with phased migration approach starting with non-production environments",
            "Implement AWS DMS for initial data synchronization", 
            "Plan for 12-16 week migration timeline with 2-week buffer",
            "Establish comprehensive testing protocols for each environment",
            "Consider Aurora for production workloads to optimize performance and cost"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"Error generating executive PDF: {e}")
        return None

def generate_technical_report_pdf_robust(results, recommendations, migration_params):
    """Generate technical report PDF - ROBUST VERSION"""
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("AWS Database Migration - Technical Report", styles['Title']))
        story.append(Spacer(1, 20))
        
        # Migration parameters
        story.append(Paragraph("Migration Configuration", styles['Heading2']))
        
        config_data = [
            ['Parameter', 'Value'],
            ['Source Engine', migration_params.get('source_engine', 'N/A')],
            ['Target Engine', migration_params.get('target_engine', 'N/A')],
            ['Data Size', f"{migration_params.get('data_size_gb', 0):,} GB"],
            ['Timeline', f"{migration_params.get('migration_timeline_weeks', 0)} weeks"],
            ['Environments', str(len(recommendations)) if recommendations else '0'],
        ]
        
        config_table = Table(config_data, colWidths=[2.5*inch, 2*inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(config_table)
        story.append(Spacer(1, 20))
        
        # Environment recommendations if available
        if recommendations:
            story.append(Paragraph("Environment Recommendations", styles['Heading2']))
            
            env_data = [['Environment', 'Instance Class', 'Configuration', 'Multi-AZ']]
            
            for env_name, rec in recommendations.items():
                if 'writer' in rec:  # Enhanced recommendations
                    instance_class = rec['writer']['instance_class']
                    config = f"Writer + {rec['readers']['count']} readers"
                    multi_az = 'Yes' if rec['writer']['multi_az'] else 'No'
                else:  # Regular recommendations
                    instance_class = rec.get('instance_class', 'N/A')
                    config = f"{rec.get('cpu_cores', 'N/A')} cores, {rec.get('ram_gb', 'N/A')} GB"
                    multi_az = 'Yes' if rec.get('multi_az', False) else 'No'
                
                env_data.append([env_name, instance_class, config, multi_az])
            
            env_table = Table(env_data)
            env_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#38a169')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            
            story.append(env_table)
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"Error generating technical PDF: {e}")
        return None

def create_bulk_reports_zip(results, recommendations, migration_params):
    """Create ZIP file with all reports"""
    
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Executive summary
            exec_pdf = generate_executive_summary_pdf_robust(results, migration_params)
            if exec_pdf:
                zip_file.writestr(
                    f"Executive_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                    exec_pdf.getvalue()
                )
            
            # Technical report
            tech_pdf = generate_technical_report_pdf_robust(results, recommendations, migration_params)
            if tech_pdf:
                zip_file.writestr(
                    f"Technical_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    tech_pdf.getvalue()
                )
            
            # CSV data
            csv_data = prepare_csv_export_data(results, recommendations)
            if csv_data is not None:
                zip_file.writestr(
                    f"Migration_Analysis_Data_{datetime.now().strftime('%Y%m%d')}.csv",
                    csv_data.to_csv(index=False)
                )
        
        zip_buffer.seek(0)
        return zip_buffer
        
    except Exception as e:
        print(f"Error creating ZIP file: {e}")
        return None

# Enhanced Environment Setup Interface
def show_enhanced_environment_setup_with_cluster_config():
    """Enhanced environment setup with Writer/Reader configuration"""
    
    st.markdown("## 📊 Enhanced Database Cluster Configuration")
    
    if not st.session_state.migration_params:
        st.warning("⚠️ Please complete Migration Configuration first.")
        return
    
    # Configuration method selection
    config_method = st.radio(
        "Choose configuration method:",
        [
            "📝 Manual Cluster Configuration", 
            "📁 Bulk Upload with Cluster Details",
            "🔄 Simple Configuration (Legacy)"
        ],
        horizontal=True
    )
    
    if config_method == "📝 Manual Cluster Configuration":
        show_manual_cluster_configuration()
    elif config_method == "📁 Bulk Upload with Cluster Details":
        show_bulk_cluster_upload()
    else:
        show_simple_configuration()

def show_manual_cluster_configuration():
    """Show manual cluster configuration with Writer/Reader options"""
    
    st.markdown("### 📝 Database Cluster Configuration")
    
    # Number of environments
    num_environments = st.number_input("Number of Environments", min_value=1, max_value=10, value=4)
    
    environment_specs = {}
    default_names = ['Development', 'QA', 'SQA', 'Production']
    
    for i in range(num_environments):
        with st.expander(f"🏢 Environment {i+1} - Cluster Configuration", expanded=i == 0):
            
            # Basic environment info
            col1, col2 = st.columns(2)
            
            with col1:
                env_name = st.text_input(
                    "Environment Name",
                    value=default_names[i] if i < len(default_names) else f"Environment_{i+1}",
                    key=f"env_name_{i}"
                )
                
                environment_type = st.selectbox(
                    "Environment Type",
                    ["Production", "Staging", "Testing", "Development"],
                    index=min(i, 3),
                    key=f"env_type_{i}"
                )
            
            with col2:
                workload_pattern = st.selectbox(
                    "Workload Pattern",
                    ["balanced", "read_heavy", "write_heavy", "analytics"],
                    key=f"workload_{i}"
                )
                
                read_write_ratio = st.slider(
                    "Read/Write Ratio (% Reads)",
                    min_value=10, max_value=95, value=70,
                    key=f"read_ratio_{i}"
                )
            
            # Infrastructure configuration
            st.markdown("#### 💻 Infrastructure Requirements")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cpu_cores = st.number_input(
                    "CPU Cores",
                    min_value=1, max_value=128,
                    value=[4, 8, 16, 32][min(i, 3)],
                    key=f"cpu_{i}"
                )
                
                ram_gb = st.number_input(
                    "RAM (GB)",
                    min_value=4, max_value=1024,
                    value=[16, 32, 64, 128][min(i, 3)],
                    key=f"ram_{i}"
                )
            
            with col2:
                storage_gb = st.number_input(
                    "Storage (GB)",
                    min_value=20, max_value=50000,
                    value=[100, 500, 1000, 2000][min(i, 3)],
                    key=f"storage_{i}"
                )
                
                iops_requirement = st.number_input(
                    "IOPS Requirement",
                    min_value=100, max_value=50000,
                    value=[1000, 3000, 5000, 10000][min(i, 3)],
                    key=f"iops_{i}"
                )
            
            with col3:
                peak_connections = st.number_input(
                    "Peak Connections",
                    min_value=1, max_value=10000,
                    value=[20, 50, 100, 500][min(i, 3)],
                    key=f"connections_{i}"
                )
                
                daily_usage_hours = st.slider(
                    "Daily Usage (Hours)",
                    min_value=1, max_value=24,
                    value=[8, 12, 16, 24][min(i, 3)],
                    key=f"usage_{i}"
                )
            
            # Cluster configuration
            st.markdown("#### 🔗 Cluster Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                multi_az_writer = st.checkbox(
                    "Multi-AZ for Writer",
                    value=environment_type in ["Production", "Staging"],
                    key=f"multi_az_writer_{i}",
                    help="Deploy writer instance across multiple Availability Zones for high availability"
                )
                
                custom_reader_count = st.checkbox(
                    "Custom Reader Count",
                    value=False,
                    key=f"custom_readers_{i}"
                )
                
                if custom_reader_count:
                    num_readers = st.number_input(
                        "Number of Read Replicas",
                        min_value=0, max_value=5,
                        value=1 if environment_type == "Production" else 0,
                        key=f"num_readers_{i}"
                    )
                else:
                    # Auto-calculate based on environment and workload
                    cluster_config = DatabaseClusterConfiguration()
                    num_readers = cluster_config.calculate_optimal_readers(
                        environment_type.lower(), workload_pattern, peak_connections
                    )
                    st.info(f"Recommended readers: {num_readers} (auto-calculated)")
            
            with col2:
                multi_az_readers = st.checkbox(
                    "Multi-AZ for Readers",
                    value=environment_type == "Production",
                    key=f"multi_az_readers_{i}",
                    help="Deploy read replicas across multiple Availability Zones"
                )
                
                if num_readers > 0:
                    custom_reader_instance = st.checkbox(
                        "Custom Reader Instance Size",
                        value=False,
                        key=f"custom_reader_instance_{i}"
                    )
                    
                    if custom_reader_instance:
                        reader_instance_override = st.selectbox(
                            "Reader Instance Class",
                            ["db.t3.medium", "db.t3.large", "db.r5.large", "db.r5.xlarge", "db.r5.2xlarge"],
                            key=f"reader_instance_{i}"
                        )
                    else:
                        reader_instance_override = None
                        st.info("Reader size will be auto-calculated based on writer size and workload")
                else:
                    reader_instance_override = None
            
            # Storage configuration
            st.markdown("#### 💾 Storage Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                storage_encrypted = st.checkbox(
                    "Encryption at Rest",
                    value=environment_type in ["Production", "Staging"],
                    key=f"encryption_{i}"
                )
            
            with col2:
                backup_retention = st.number_input(
                    "Backup Retention (Days)",
                    min_value=1, max_value=35,
                    value=30 if environment_type == "Production" else 7,
                    key=f"backup_{i}"
                )
            
            with col3:
                auto_storage_scaling = st.checkbox(
                    "Auto Storage Scaling",
                    value=True,
                    key=f"auto_scaling_{i}",
                    help="Automatically scale storage when needed"
                )
            
            # Store environment configuration
            environment_specs[env_name] = {
                'cpu_cores': cpu_cores,
                'ram_gb': ram_gb,
                'storage_gb': storage_gb,
                'iops_requirement': iops_requirement,
                'peak_connections': peak_connections,
                'daily_usage_hours': daily_usage_hours,
                'workload_pattern': workload_pattern,
                'read_write_ratio': read_write_ratio,
                'environment_type': environment_type.lower(),
                'multi_az_writer': multi_az_writer,
                'multi_az_readers': multi_az_readers,
                'num_readers': num_readers if custom_reader_count else None,
                'reader_instance_override': reader_instance_override,
                'storage_encrypted': storage_encrypted,
                'backup_retention': backup_retention,
                'auto_storage_scaling': auto_storage_scaling
            }
    
    if st.button("💾 Save Cluster Configuration", type="primary", use_container_width=True):
        st.session_state.environment_specs = environment_specs
        st.success("✅ Cluster configuration saved!")
        
        # Run enhanced analysis
        if st.button("🚀 Analyze Cluster Configuration", type="secondary", use_container_width=True):
            with st.spinner("🔄 Analyzing cluster requirements..."):
                analyzer = EnhancedMigrationAnalyzer()
                recommendations = analyzer.calculate_enhanced_instance_recommendations(environment_specs)
                st.session_state.enhanced_recommendations = recommendations
                
                # Show preview
                show_cluster_configuration_preview(recommendations)

def show_bulk_cluster_upload():
    """Show bulk upload interface for cluster configurations"""
    
    st.markdown("### 📁 Bulk Cluster Configuration Upload")
    
    with st.expander("📋 Download Enhanced Template", expanded=False):
        st.markdown("""
        **Enhanced Template includes:**
        - Writer/Reader configuration
        - Multi-AZ options
        - Storage specifications
        - Workload patterns
        - IOPS requirements
        """)
        
        # Generate enhanced template
        enhanced_template = create_enhanced_cluster_template()
        csv_data = enhanced_template.to_csv(index=False)
        
        st.dataframe(enhanced_template, use_container_width=True)
        
        st.download_button(
            label="📥 Download Enhanced Cluster Template",
            data=csv_data,
            file_name="enhanced_cluster_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Cluster Configuration",
        type=['csv', 'xlsx'],
        help="Upload CSV or Excel file with cluster specifications"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Process cluster data
            environment_specs = process_cluster_data(df)
            
            if environment_specs:
                st.session_state.environment_specs = environment_specs
                st.success(f"✅ Successfully processed {len(environment_specs)} cluster configurations!")
                
                # Show summary
                show_cluster_upload_summary(environment_specs)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def create_enhanced_cluster_template() -> pd.DataFrame:
    """Create enhanced cluster template"""
    
    template_data = {
        'Environment_Name': ['Production-Cluster', 'Staging-Cluster', 'QA-Cluster', 'Dev-Cluster'],
        'Environment_Type': ['Production', 'Staging', 'Testing', 'Development'],
        'CPU_Cores': [32, 16, 8, 4],
        'RAM_GB': [128, 64, 32, 16],
        'Storage_GB': [2000, 1000, 500, 200],
        'IOPS_Requirement': [10000, 5000, 3000, 1000],
        'Peak_Connections': [500, 200, 100, 50],
        'Daily_Usage_Hours': [24, 16, 12, 8],
        'Workload_Pattern': ['read_heavy', 'balanced', 'balanced', 'write_heavy'],
        'Read_Write_Ratio': [80, 70, 60, 40],
        'Multi_AZ_Writer': [True, True, False, False],
        'Multi_AZ_Readers': [True, False, False, False],
        'Num_Readers': [2, 1, 1, 0],
        'Storage_Encrypted': [True, True, False, False],
        'Backup_Retention_Days': [30, 14, 7, 7],
        'Auto_Storage_Scaling': [True, True, True, False]
    }
    
    return pd.DataFrame(template_data)

def process_cluster_data(df: pd.DataFrame) -> Dict:
    """Process uploaded cluster data"""
    
    environments = {}
    
    for _, row in df.iterrows():
        env_name = str(row['Environment_Name'])
        
        environments[env_name] = {
            'cpu_cores': int(row.get('CPU_Cores', 4)),
            'ram_gb': int(row.get('RAM_GB', 16)),
            'storage_gb': int(row.get('Storage_GB', 500)),
            'iops_requirement': int(row.get('IOPS_Requirement', 3000)),
            'peak_connections': int(row.get('Peak_Connections', 100)),
            'daily_usage_hours': int(row.get('Daily_Usage_Hours', 24)),
            'workload_pattern': str(row.get('Workload_Pattern', 'balanced')),
            'read_write_ratio': int(row.get('Read_Write_Ratio', 70)),
            'environment_type': str(row.get('Environment_Type', 'Production')).lower(),
            'multi_az_writer': bool(row.get('Multi_AZ_Writer', True)),
            'multi_az_readers': bool(row.get('Multi_AZ_Readers', False)),
            'num_readers': int(row.get('Num_Readers', 1)) if pd.notna(row.get('Num_Readers')) else None,
            'storage_encrypted': bool(row.get('Storage_Encrypted', True)),
            'backup_retention': int(row.get('Backup_Retention_Days', 7)),
            'auto_storage_scaling': bool(row.get('Auto_Storage_Scaling', True))
        }
    
    return environments

def show_cluster_upload_summary(environment_specs: Dict):
    """Show summary of uploaded cluster configurations"""
    
    st.markdown("#### 📊 Cluster Configuration Summary")
    
    summary_data = []
    for env_name, specs in environment_specs.items():
        summary_data.append({
            'Environment': env_name,
            'Type': specs['environment_type'].title(),
            'Resources': f"{specs['cpu_cores']} cores, {specs['ram_gb']} GB RAM",
            'Storage': f"{specs['storage_gb']} GB ({specs['iops_requirement']} IOPS)",
            'Workload': f"{specs['workload_pattern']} ({specs['read_write_ratio']}% reads)",
            'Writer Multi-AZ': '✅' if specs['multi_az_writer'] else '❌',
            'Readers': f"{specs.get('num_readers', 'Auto')} ({'Multi-AZ' if specs['multi_az_readers'] else 'Single-AZ'})"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

def show_cluster_configuration_preview(recommendations: Dict):
    """Show preview of cluster configuration recommendations"""
    
    st.markdown("#### 🎯 Cluster Configuration Preview")
    
    for env_name, rec in recommendations.items():
        with st.expander(f"🏢 {env_name} Cluster Configuration"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Writer Configuration**")
                st.write(f"Instance: {rec['writer']['instance_class']}")
                st.write(f"Multi-AZ: {'✅ Yes' if rec['writer']['multi_az'] else '❌ No'}")
                st.write(f"Resources: {rec['writer']['cpu_cores']} cores, {rec['writer']['ram_gb']} GB RAM")
            
            with col2:
                st.markdown("**Reader Configuration**")
                reader_count = rec['readers']['count']
                if reader_count > 0:
                    st.write(f"Count: {reader_count} replicas")
                    st.write(f"Instance: {rec['readers']['instance_class']}")
                    st.write(f"Multi-AZ: {'✅ Yes' if rec['readers']['multi_az'] else '❌ No'}")
                else:
                    st.write("No read replicas")
                    st.write("Single writer configuration")
            
            with col3:
                st.markdown("**Storage Configuration**")
                storage = rec['storage']
                st.write(f"Size: {storage['size_gb']} GB")
                st.write(f"Type: {storage['type'].upper()}")
                st.write(f"IOPS: {storage['iops']:,}")
                st.write(f"Encrypted: {'✅ Yes' if storage['encrypted'] else '❌ No'}")

# Enhanced Cost Analysis Functions
def show_enhanced_cost_analysis():
    """Show enhanced cost analysis with Writer/Reader breakdown"""
    
    st.markdown("### 💰 Enhanced Cost Analysis")
    
    if not hasattr(st.session_state, 'enhanced_analysis_results'):
        st.warning("Please run the enhanced analysis first.")
        return
    
    results = st.session_state.enhanced_analysis_results
    
    # Initialize chart counter to avoid undefined variable error
    chart_counter = 0
    
    # Overall cost metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Monthly Cost",
            f"${results['monthly_aws_cost']:,.0f}",
            delta=f"${results['annual_aws_cost']:,.0f}/year"
        )
    
    with col2:
        # Calculate writer vs reader costs
        total_writer_cost = sum([env.get('writer_instance_cost', 0) for env in results['environment_costs'].values()])
        total_reader_cost = sum([env.get('reader_costs', 0) for env in results['environment_costs'].values()])
        
        st.metric(
            "Writer Instances",
            f"${total_writer_cost:,.0f}/month",
            delta=f"{(total_writer_cost/results['monthly_aws_cost']*100):.1f}% of total"
        )
    
    with col3:
        st.metric(
            "Reader Instances",
            f"${total_reader_cost:,.0f}/month",
            delta=f"{(total_reader_cost/results['monthly_aws_cost']*100):.1f}% of total" if total_reader_cost > 0 else "No readers"
        )
    
    with col4:
        total_storage_cost = sum([env.get('storage_cost', 0) for env in results['environment_costs'].values()])
        st.metric(
            "Storage & I/O",
            f"${total_storage_cost:,.0f}/month",
            delta=f"{(total_storage_cost/results['monthly_aws_cost']*100):.1f}% of total"
        )
    
    # Detailed environment breakdown
    st.markdown("#### 🏢 Environment Cost Breakdown")
    
    for env_name, costs in results['environment_costs'].items():
        with st.expander(f"💵 {env_name} - Total: ${costs.get('total_monthly', 0):,.0f}/month"):
            
            # Create cost breakdown chart with safe data access
            cost_categories = ['Writer Instance', 'Reader Instances', 'Storage', 'Backup', 'Monitoring', 'Cross-AZ Transfer']
            cost_values = [
                costs.get('writer_instance_cost', 0),
                costs.get('reader_costs', 0),
                costs.get('storage_cost', 0),
                costs.get('backup_cost', 0),
                costs.get('monitoring_cost', 0),
                costs.get('cross_az_cost', 0)
            ]
            
            fig = go.Figure(data=[go.Pie(
                labels=cost_categories,
                values=cost_values,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title=f"{env_name} Cost Distribution",
                height=400
            )
            
            # FIXED: Use unique key with environment name
            st.plotly_chart(fig, use_container_width=True, key=f"enhanced_cost_breakdown_{env_name}_{chart_counter}")
            chart_counter += 1


# Enhanced Analysis Runner
def run_enhanced_migration_analysis():
    """Run enhanced migration analysis with Writer/Reader support"""
    
    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedMigrationAnalyzer()
        
        # Step 1: Calculate enhanced recommendations
        st.write("📊 Calculating cluster recommendations...")
        recommendations = analyzer.calculate_enhanced_instance_recommendations(st.session_state.environment_specs)
        st.session_state.enhanced_recommendations = recommendations
        
        # Step 2: Calculate enhanced costs
        st.write("💰 Analyzing cluster costs...")
        cost_analysis = analyzer.calculate_enhanced_migration_costs(recommendations, st.session_state.migration_params)
        st.session_state.enhanced_analysis_results = cost_analysis
        
        # Step 3: Generate cost comparison
        st.write("📈 Generating cost comparisons...")
        generate_enhanced_cost_visualizations()
        
        st.success("✅ Enhanced analysis complete!")
        
        # Show summary
        show_enhanced_analysis_summary()
        
    except Exception as e:
        st.error(f"❌ Enhanced analysis failed: {str(e)}")
        st.code(str(e))

def show_enhanced_analysis_summary():
    """Show enhanced analysis summary"""
    
    st.markdown("#### 🎯 Enhanced Analysis Summary")
    
    results = st.session_state.enhanced_analysis_results
    recommendations = st.session_state.enhanced_recommendations
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_writers = len(recommendations)
    total_readers = sum([rec['readers']['count'] for rec in recommendations.values()])
    multi_az_envs = sum([1 for rec in recommendations.values() if rec['writer']['multi_az']])
    
    with col1:
        st.metric("Total Environments", total_writers)
    
    with col2:
        st.metric("Total Read Replicas", total_readers)
    
    with col3:
        st.metric("Multi-AZ Environments", multi_az_envs)
    
    with col4:
        avg_monthly_cost = results['monthly_aws_cost'] / total_writers
        st.metric("Avg Cost per Environment", f"${avg_monthly_cost:,.0f}/month")
    
    # Configuration overview
    st.markdown("#### 📋 Configuration Overview")
    
    config_summary = []
    for env_name, rec in recommendations.items():
        config_summary.append({
            'Environment': env_name,
            'Writer': f"{rec['writer']['instance_class']} ({'Multi-AZ' if rec['writer']['multi_az'] else 'Single-AZ'})",
            'Readers': f"{rec['readers']['count']} x {rec['readers']['instance_class']}" if rec['readers']['count'] > 0 else "None",
            'Storage': f"{rec['storage']['size_gb']} GB {rec['storage']['type'].upper()}",
            'IOPS': f"{rec['storage']['iops']:,}",
            'Workload': f"{rec['workload_pattern']} ({rec['read_write_ratio']}% reads)"
        })
    
    config_df = pd.DataFrame(config_summary)
    st.dataframe(config_df, use_container_width=True)

def generate_enhanced_cost_visualizations():
    """Generate enhanced cost visualizations"""
    
    results = st.session_state.enhanced_analysis_results
    
    # Writer vs Reader cost comparison
    env_names = list(results['environment_costs'].keys())
    writer_costs = [results['environment_costs'][env]['writer_instance_cost'] for env in env_names]
    reader_costs = [results['environment_costs'][env]['reader_costs'] for env in env_names]
    storage_costs = [results['environment_costs'][env]['storage_cost'] for env in env_names]
    
# Create stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Writer Instance', x=env_names, y=writer_costs, marker_color='#3182ce'))
    fig.add_trace(go.Bar(name='Reader Instances', x=env_names, y=reader_costs, marker_color='#38a169'))
    fig.add_trace(go.Bar(name='Storage & I/O', x=env_names, y=storage_costs, marker_color='#d69e2e'))
    
    fig.update_layout(
        title='Monthly Cost Breakdown by Environment',
        xaxis_title='Environment',
        yaxis_title='Monthly Cost ($)',
        barmode='stack',
        height=500
    )
    
    st.session_state.enhanced_cost_chart = fig

# Integration with main application
def integrate_enhanced_cluster_features():
    """Integration instructions for enhanced cluster features"""
    
    # Replace show_environment_setup() with show_enhanced_environment_setup_with_cluster_config()
    # Replace run_migration_analysis() with run_enhanced_migration_analysis()
    # Add show_enhanced_cost_analysis() to the results dashboard
    
    # Add to session state initialization:
    # 'enhanced_recommendations': None,
    # 'enhanced_analysis_results': None,
    # 'enhanced_cost_chart': None
    
    pass

# ADD this helper function to check data compatibility:

def export_recommendations_to_csv(optimization_results):
    """Export optimization results to CSV format"""
    
    export_data = []
    
    for env_name, optimization in optimization_results.items():
        writer = optimization['writer_optimization']
        reader = optimization['reader_optimization']
        costs = optimization['cost_analysis']['monthly_breakdown']
        
        export_data.append({
            'Environment': env_name,
            'Environment_Type': optimization['environment_type'],
            'Optimization_Score': f"{optimization['optimization_score']:.1f}",
            'Writer_Instance': writer['instance_class'],
            'Writer_vCPUs': writer['specs'].vcpu,
            'Writer_Memory_GB': writer['specs'].memory_gb,
            'Writer_Monthly_Cost': f"${writer['monthly_cost']:,.0f}",
            'Writer_Annual_Cost': f"${writer['annual_cost']:,.0f}",
            'Reader_Count': reader['count'],
            'Reader_Instance': reader['instance_class'] if reader['count'] > 0 else 'None',
            'Reader_Monthly_Cost_Total': f"${reader['total_monthly_cost']:,.0f}",
            'Reader_Annual_Cost_Total': f"${reader['total_annual_cost']:,.0f}",
            'Storage_Monthly_Cost': f"${costs['storage']:,.0f}",
            'Total_Monthly_Cost': f"${costs['total']:,.0f}",
            'Total_Annual_Cost': f"${costs['total'] * 12:,.0f}",
            'Reserved_1Y_Savings': f"${optimization['cost_analysis']['reserved_instance_options']['1_year']['total_savings']:,.0f}",
            'Reserved_3Y_Savings': f"${optimization['cost_analysis']['reserved_instance_options']['3_year']['total_savings']:,.0f}",
            'Workload_Type': optimization['workload_analysis']['workload_type'],
            'Complexity_Score': f"{optimization['workload_analysis']['complexity_score']:.1f}",
            'Writer_Reasoning': writer['reasoning'],
            'Reader_Reasoning': reader['reasoning']
        })
    
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False)

def generate_optimization_report_pdf(optimization_results):
    """Generate detailed PDF report for optimization results"""
    
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    import io
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("Reader/Writer Optimization Report", styles['Title']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    total_monthly = sum([opt['cost_analysis']['monthly_breakdown']['total'] for opt in optimization_results.values()])
    avg_score = sum([opt['optimization_score'] for opt in optimization_results.values()]) / len(optimization_results)
    
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Paragraph(f"Total Environments Analyzed: {len(optimization_results)}", styles['Normal']))
    story.append(Paragraph(f"Total Monthly Cost: ${total_monthly:,.0f}", styles['Normal']))
    story.append(Paragraph(f"Total Annual Cost: ${total_monthly * 12:,.0f}", styles['Normal']))
    story.append(Paragraph(f"Average Optimization Score: {avg_score:.1f}/100", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Detailed recommendations for each environment
    for env_name, optimization in optimization_results.items():
        story.append(Paragraph(f"{env_name} Environment", styles['Heading2']))
        
        # Create summary table
        writer = optimization['writer_optimization']
        reader = optimization['reader_optimization']
        
        env_data = [
            ['Component', 'Configuration', 'Monthly Cost', 'Annual Cost'],
            ['Writer Instance', 
             f"{writer['instance_class']} ({writer['specs'].vcpu} vCPU, {writer['specs'].memory_gb}GB)",
             f"${writer['monthly_cost']:,.0f}",
             f"${writer['annual_cost']:,.0f}"],
            ['Read Replicas',
             f"{reader['count']} x {reader['instance_class']}" if reader['count'] > 0 else "None",
             f"${reader['total_monthly_cost']:,.0f}",
             f"${reader['total_annual_cost']:,.0f}"],
            ['Total Environment',
             f"Score: {optimization['optimization_score']:.1f}/100",
             f"${optimization['cost_analysis']['monthly_breakdown']['total']:,.0f}",
             f"${optimization['cost_analysis']['monthly_breakdown']['total'] * 12:,.0f}"]
        ]
        
        env_table = Table(env_data, colWidths=[1.5*inch, 2.5*inch, 1.2*inch, 1.2*inch])
        env_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        
        story.append(env_table)
        story.append(Spacer(1, 15))
        
        # Add reasoning
        story.append(Paragraph("Writer Reasoning:", styles['Heading3']))
        story.append(Paragraph(writer['reasoning'], styles['Normal']))
        story.append(Paragraph("Reader Reasoning:", styles['Heading3']))
        story.append(Paragraph(reader['reasoning'], styles['Normal']))
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Step 5: Add optimization preferences to the OptimizedReaderWriterAnalyzer class
# Add this method to the OptimizedReaderWriterAnalyzer class:

def set_optimization_preferences(self, goal='balanced', consider_reserved=True, environment_priority='production_first'):
    """Set optimization preferences"""
    self.optimization_goal = goal
    self.consider_reserved_instances = consider_reserved
    self.environment_priority = environment_priority
    
    # Adjust scoring weights based on goal
    if goal == 'cost_efficiency':
        self.performance_weight = 0.2
        self.cost_weight = 0.6
        self.suitability_weight = 0.2
    elif goal == 'performance':
        self.performance_weight = 0.7
        self.cost_weight = 0.1
        self.suitability_weight = 0.2
    else:  # balanced
        self.performance_weight = 0.4
        self.cost_weight = 0.3
        self.suitability_weight = 0.3

def is_enhanced_environment_data(environment_specs):
    """Check if environment specs contain enhanced cluster data"""
    if not environment_specs:
        return False
    
    # Check if any environment has enhanced fields
    sample_spec = next(iter(environment_specs.values()))
    enhanced_fields = ['workload_pattern', 'read_write_ratio', 'multi_az_writer', 'environment_type']
    
    return any(field in sample_spec for field in enhanced_fields)


def show_enhanced_environment_analysis():
    """Show enhanced environment analysis with Writer/Reader details"""
    
    st.markdown("### 🏢 Enhanced Environment Analysis")
    
    if not hasattr(st.session_state, 'enhanced_recommendations') or not st.session_state.enhanced_recommendations:
        st.warning("Enhanced recommendations not available.")
        return
    
    recommendations = st.session_state.enhanced_recommendations
    environment_specs = st.session_state.environment_specs
    
    # Environment comparison with cluster details
    env_comparison_data = []
    
    for env_name, rec in recommendations.items():
        specs = environment_specs.get(env_name, {})
        
        # Writer configuration
        writer_config = f"{rec['writer']['instance_class']} ({'Multi-AZ' if rec['writer']['multi_az'] else 'Single-AZ'})"
        
        # Reader configuration
        reader_count = rec['readers']['count']
        if reader_count > 0:
            reader_config = f"{reader_count} x {rec['readers']['instance_class']}"
        else:
            reader_config = "No readers"
        
        env_comparison_data.append({
            'Environment': env_name,
            'Type': rec['environment_type'].title(),
            'Current Resources': f"{specs.get('cpu_cores', 'N/A')} cores, {specs.get('ram_gb', 'N/A')} GB RAM",
            'Writer Instance': writer_config,
            'Read Replicas': reader_config,
            'Storage': f"{rec['storage']['size_gb']} GB {rec['storage']['type'].upper()}",
            'Workload Pattern': f"{rec['workload_pattern']} ({rec['read_write_ratio']}% reads)"
        })
    
    env_df = pd.DataFrame(env_comparison_data)
    st.dataframe(env_df, use_container_width=True)
    
    # Detailed environment insights
    st.markdown("#### 💡 Environment Insights")
    
    for env_name, rec in recommendations.items():
        with st.expander(f"🔍 {env_name} Environment Details"):
            specs = environment_specs.get(env_name, {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Current Configuration**")
                st.write(f"CPU Cores: {specs.get('cpu_cores', 'N/A')}")
                st.write(f"RAM: {specs.get('ram_gb', 'N/A')} GB")
                st.write(f"Storage: {specs.get('storage_gb', 'N/A')} GB")
                st.write(f"IOPS Requirement: {specs.get('iops_requirement', 'N/A')}")
                st.write(f"Peak Connections: {specs.get('peak_connections', 'N/A')}")
            
            with col2:
                st.markdown("**Writer Configuration**")
                writer = rec['writer']
                st.write(f"Instance: {writer['instance_class']}")
                st.write(f"Multi-AZ: {'Yes' if writer['multi_az'] else 'No'}")
                st.write(f"CPU Cores: {writer['cpu_cores']}")
                st.write(f"RAM: {writer['ram_gb']} GB")
                
                st.markdown("**Reader Configuration**")
                readers = rec['readers']
                if readers['count'] > 0:
                    st.write(f"Count: {readers['count']}")
                    st.write(f"Instance: {readers['instance_class']}")
                    st.write(f"Multi-AZ: {'Yes' if readers['multi_az'] else 'No'}")
                else:
                    st.write("No read replicas")
            
            with col3:
                st.markdown("**Storage Configuration**")
                storage = rec['storage']
                st.write(f"Size: {storage['size_gb']} GB")
                st.write(f"Type: {storage['type'].upper()}")
                st.write(f"IOPS: {storage['iops']:,}")
                st.write(f"Encrypted: {'Yes' if storage['encrypted'] else 'No'}")
                st.write(f"Backup Retention: {storage['backup_retention_days']} days")
                
                st.markdown("**Workload Characteristics**")
                st.write(f"Pattern: {rec['workload_pattern']}")
                st.write(f"Read/Write Ratio: {rec['read_write_ratio']}% reads")
                st.write(f"Peak Connections: {rec['connections']}")
            
            # Optimization recommendations
            st.markdown("**💡 Optimization Notes**")
            
            if rec['environment_type'] == 'production':
                st.success("✅ Production-grade configuration with high availability")
                if readers['count'] > 0:
                    st.info(f"📊 {readers['count']} read replicas will help distribute read load")
            elif rec['environment_type'] == 'development':
                st.info("💡 Cost-optimized configuration for development")
                if readers['count'] == 0:
                    st.info("💰 No read replicas to minimize development costs")
            
            if rec['workload_pattern'] == 'read_heavy' and readers['count'] > 0:
                st.success(f"📈 Read-heavy workload well-suited for {readers['count']} read replicas")
            elif rec['workload_pattern'] == 'read_heavy' and readers['count'] == 0:
                st.warning("⚠️ Read-heavy workload might benefit from read replicas")
            
            if storage['type'] == 'io2':
                st.info("⚡ High-performance io2 storage for demanding IOPS requirements")
            elif storage['type'] == 'gp3':
                st.info("⚖️ Balanced gp3 storage for general-purpose workloads")
def validate_growth_analysis_functions():
    """Validate that all growth analysis functions are properly defined"""
    
    required_functions = [
        'show_basic_cost_summary',
        'show_growth_analysis_dashboard', 
        'show_risk_assessment_tab',
        'show_environment_analysis_tab',
        'show_visualizations_tab',
        'show_ai_insights_tab',
        'show_timeline_analysis_tab',
        'create_growth_projection_charts'
    ]
    
    missing_functions = []
    
    for func_name in required_functions:
        if func_name not in globals():
            missing_functions.append(func_name)
    
    if missing_functions:
        st.error(f"❌ Missing functions: {', '.join(missing_functions)}")
        return False
    else:
        st.success("✅ All growth analysis functions are properly defined!")
        return True

# Add this to your main function to test
def test_growth_setup():
    """Test the growth analysis setup"""
    st.markdown("### 🧪 Growth Analysis Setup Test")
    
    if st.button("🔍 Validate Setup"):
        validate_growth_analysis_functions()
        
        # Test growth analyzer
        try:
            analyzer = GrowthAwareCostAnalyzer()
            st.success("✅ GrowthAwareCostAnalyzer initialized successfully!")
        except Exception as e:
            st.error(f"❌ GrowthAwareCostAnalyzer error: {str(e)}")
        
        # Test session state
        if hasattr(st.session_state, 'growth_analysis'):
            st.info("📊 Growth analysis data available in session state")
        else:
            st.warning("⚠️ No growth analysis data in session state yet")




if __name__ == "__main__":
    main()