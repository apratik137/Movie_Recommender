import logging
from model_SVD import prediction
import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"{prediction(int(name))}")
    else:
        return func.HttpResponse(
             f"{prediction(1234)}",
             status_code=200
             
        )
