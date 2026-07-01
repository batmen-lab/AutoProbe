# ieee_cis_fraud_detection — XGBoost fraud (AUPRC)

Echelon (baseline 0.1267 -> best **0.4663**, ROC-AUC 0.861):
  r2 0.15 (+0.023) · r3 0.4349 (+0.285, features 8->30) · r6 0.4663 (+0.031, depth 2->6 + stratified split)
=> 3 take-offs; rounds 7-10 plateaued (reverted) at 0.4663.
final_train.py is the best (round-6) XGBoost model.
