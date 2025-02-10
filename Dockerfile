FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install poetry==1.4.2

WORKDIR /frame_classification

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry install --without dev

FROM python:3.9

# Install pytest
RUN pip install pytest
RUN pip install av

# Copy the rest of the project files
COPY . .

# Run tests using Poetry
CMD ["poetry", "run", "pytest", "tests/"]
