@echo off
title Instalador Removedor de Marcas dagua
color 0A

echo ==============================================================================
echo      REMOVEDOR DE MARCAS D'AGUA - IA (SETUP AUTOMATICO)
echo ==============================================================================
echo.

REM 1. Verifica se o Python existe
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERRO CRITICO] Python nao encontrado!
    echo Por favor, instale o Python 3.10 ou 3.11 no site python.org
    echo Importante: Marque a caixa "Add Python to PATH" na instalacao.
    pause
    exit
)

REM 2. Cria o ambiente virtual (venv) se nao existir
if not exist "venv" (
    echo [SETUP] Criando ambiente virtual isolado...
    python -m venv venv
)

REM 3. Define variaveis para usar o Python do venv
set PYTHON_CMD=venv\Scripts\python.exe
set PIP_CMD=venv\Scripts\pip.exe

REM 4. Verifica se as dependencias ja foram instaladas
if not exist "venv\instalado.lock" (
    echo.
    echo [SETUP] Primeira execucao detectada. Instalando bibliotecas...
    echo Isso pode levar alguns minutos (Download de ~3GB de IA).
    echo.
    
    echo [1/4] Atualizando gerenciador de pacotes...
    %PYTHON_CMD% -m pip install --upgrade pip

    echo.
    echo [2/4] Instalando PyTorch com suporte a NVIDIA CUDA (GPU)...
    REM Instala a versao especifica que voce usava (2.5.1+cu121)
    %PIP_CMD% install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

    echo.
    echo [3/4] Instalando bibliotecas do projeto...
    %PIP_CMD% install -r requirements.txt

    echo.
    echo [4/4] Finalizando configuracao...
    echo Instalacao Concluida > venv\instalado.lock
    cls
    echo [SUCESSO] Ambiente configurado com sucesso!
)

REM 5. Inicia o programa
echo.
echo Iniciando aplicacao...
%PYTHON_CMD% removedor.py

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERRO] O programa fechou com erro. Veja a mensagem acima.
    pause
)