# Real AWS verification checklist

The repository's mock-based unit tests do not establish successful AWS usage.

1. In the AWS console, verify an active account, region, budget alert, and model access.
2. Use an IAM principal with the least privileges necessary for `bedrock:InvokeModel` on the selected model.
3. Set `BEDROCK_MODEL_ID` and `AWS_REGION` in your shell (or configure a secure environment).
4. Start Streamlit, enter a non-sensitive prompt, verify a genuine response.
5. Capture a redacted screenshot and record the model ID, date, region and approximate latency.
6. For your résumé, describe only the actual work completed and verified.
