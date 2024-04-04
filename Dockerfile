FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workdir
COPY . /workdir

RUN pip install jupyter notebook

RUN pip install -e

EXPOSE 8003

CMD ["jupyter", "notebook", "--ip='*'", "--port=8003", "--no-browser", "--allow-root"]