import json
import boto3

def lambda_handler(event, context):
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    body = json.loads(event['body'])
    
    headline = body['query']['headline']
    
    # headline = 'SpaceX-Polaris crew attempts first private spacewalk'
    
    endpoint_name = 'multiclass-text-classification-endpoint-final-v1'
    
    payload = json.dumps({"inputs": headline})
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName = endpoint_name,
        ContentType = "application/json",
        Body = payload
        )
    result = json.loads(response['Body'].read().decode())
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }