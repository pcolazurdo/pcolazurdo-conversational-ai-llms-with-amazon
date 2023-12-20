set -e

AWS_CLI=aws
CURRENT_PATH=`pwd`
source ./build_langchain_layer.sh
S3_ASSETS_BUCKET=pabcol-us-east-1
VERSION=v1
VERSION_PREFIX=artifacts/ML-12016/$VERSION

echo "Packaging Lambda"
cd ./src/bot_dispatcher

zip -v -r9 ${CURRENT_PATH}/build/lex-flan-lambda.zip ./dispatchers/ ./sm_utils/ __init__.py lex_langchain_hook_function.py
$AWS_CLI s3 cp ${CURRENT_PATH}/build/lex-flan-lambda.zip s3://$S3_ASSETS_BUCKET/$VERSION_PREFIX/lex-flan-lambda.zip 
$AWS_CLI s3 cp ${CURRENT_PATH}/build/langchain_layer.zip s3://$S3_ASSETS_BUCKET/$VERSION_PREFIX/langchain_layer.zip 

echo "Uploading Static CFN"

JS_LAMBDA_HOOK_CFN=SMJumpstartFlanT5-LambdaHook.template.json

# cd ${CURRENT_PATH}/static
# JS_LAMBDA_HOOK_DST=s3://$S3_ASSETS_BUCKET/$VERSION_PREFIX/stacks/$JS_LAMBDA_HOOK_CFN
# $AWS_CLI s3 cp ./$JS_LAMBDA_HOOK_CFN $JS_LAMBDA_HOOK_DST 

cd ${CURRENT_PATH}/static
aws cloudformation create-stack --stack-name ConversationBot --template-body file://${JS_LAMBDA_HOOK_CFN} --parameters ParameterKey=Version,ParameterValue=artifacts/ML-12016/v1 ParameterKey=S3BucketName,ParameterValue=pabcol-us-east-1 --capabilities CAPABILITY_NAMED_IAM --region us-east-1
cd ${CURRENT_PATH}
