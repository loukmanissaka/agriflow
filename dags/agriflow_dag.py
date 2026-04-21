from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import random

# Fonction simulant ton Equation 1 (Score local)
def calculate_source_quality(source_name):
    score = random.uniform(0.88, 0.99) # Emulation de tes résultats Monte Carlo
    print(f"Source {source_name} - Qualité locale q: {score:.3f}")
    return score

# Fonction simulant ton Equation 6 (Propagation + Porte)
def quality_gate_logic(**kwargs):
    ti = kwargs['ti']
    q_iot = ti.xcom_pull(task_ids='ingest_iot')
    q_sat = ti.xcom_pull(task_ids='fetch_sentinel')
    
    # Ton calcul AgriFlow : Q_j = q_local * (Q_i ^ w)
    Q_final = 0.96 * (q_iot**0.5 * q_sat**0.5)
    
    if Q_final < 0.85:
        raise ValueError(f"CRITIQUE : Qualité {Q_final:.3f} sous le seuil !")
    print(f"Validation AgriFlow réussie : Q = {Q_final:.3f}")

with DAG(
    'AgriFlow_Operational_v1',
    start_date=datetime(2026, 1, 1),
    schedule_interval='@hourly', # Pour simuler la haute fréquence
    catchup=False
) as dag:

    t1 = PythonOperator(task_id='ingest_iot', python_callable=calculate_source_quality, op_args=['IoT'])
    t2 = PythonOperator(task_id='fetch_sentinel', python_callable=calculate_source_quality, op_args=['Sat'])
    
    # La fameux Quality Gate de ta Config E
    t3 = PythonOperator(task_id='agriflow_quality_gate', python_callable=quality_gate_logic)

    [t1, t2] >> t3
