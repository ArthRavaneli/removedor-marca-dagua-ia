@echo off
title Removedor de Marcas d'Agua (IA Local)
color 0B
mode con: cols=100 lines=30

:INICIO
cls
echo.
echo  ==================================================================================================
echo   REMOVEDOR AUTOMATICO DE MARCAS D'AGUA (v69)
echo   Tecnologia: Florence-2 + LaMA Inpainting
echo  ==================================================================================================
echo.

REM --- 1. VERIFICACAO DO PYTHON ---
echo  [1/4] Verificando instalacao do Python...
python --version >nul 2>&1
if %errorlevel% neq 0 goto ERRO_PYTHON
echo        Python detectado com sucesso.

REM --- 2. VERIFICACAO DO AMBIENTE VIRTUAL ---
echo.
echo  [2/4] Verificando Ambiente Virtual (VENV)...
if exist "venv" goto VENV_PRONTO
echo        Criando ambiente isolado (pode levar alguns segundos)...
python -m venv venv
:VENV_PRONTO
set PYTHON_CMD=venv\Scripts\python.exe
set PIP_CMD=venv\Scripts\pip.exe
echo        Ambiente pronto.

REM --- 3. VERIFICACAO DE DEPENDENCIAS ---
echo.
echo  [3/4] Verificando bibliotecas e motores de IA...

if exist "venv\instalado.lock" goto PULAR_INSTALACAO

REM --- BLOCO DE INSTALACAO (Primeira Vez) ---
color 0E
cls
echo.
echo  ==================================================================================================
echo   PRIMEIRA EXECUCAO DETECTADA - INSTALACAO DE FABRICA
echo  ==================================================================================================
echo.
echo   Avisos Importantes:
echo   1. O sistema vai baixar cerca de 3GB de arquivos (PyTorch, CUDA, Modelos).
echo   2. Dependendo da sua internet, isso pode levar de 5 a 20 minutos.
echo   3. NAO FECHE ESTA JANELA enquanto os downloads estiverem rolando.
echo.
echo   [PASSO 1] Atualizando pip...
%PYTHON_CMD% -m pip install --upgrade pip

echo.
echo   [PASSO 2] Baixando Bibliotecas Gerais...
%PIP_CMD% install -r requirements.txt

echo.
echo   [PASSO 3] Configurando Aceleracao de GPU (NVIDIA CUDA)...
%PIP_CMD% install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps

echo.
echo   [SUCESSO] Instalacao finalizada.
echo   Instalacao Concluida > venv\instalado.lock
timeout /t 3 >nul
color 0B
cls
goto INICIO

:PULAR_INSTALACAO
echo        Bibliotecas ja instaladas. Pulando verificacao.

REM --- 4. EXECUCAO DO PROGRAMA ---
echo.
echo  ==================================================================================================
echo   [4/4] INICIALIZANDO SOFTWARE
echo  ==================================================================================================
echo.
echo   STATUS: Carregando modelos de Inteligencia Artificial na memoria...
echo   NOTA:   Isso pode levar de 10 a 30 segundos. A janela abrira automaticamente.
echo.
echo   Por favor, aguarde...

REM Executa o script
%PYTHON_CMD% removedor.py

REM --- 5. VERIFICACAO DE ERRO ---
if %errorlevel% neq 0 goto ERRO_EXECUCAO

echo.
echo   Programa encerrado pelo usuario.
timeout /t 3 >nul
exit

:ERRO_PYTHON
color 0C
echo.
echo  [ERRO CRITICO] Python nao encontrado!
echo.
echo  1. Baixe o Python 3.10 ou 3.11 em python.org
echo  2. Na instalacao, MARQUE a opcao "Add Python to PATH"
echo.
pause
exit

:ERRO_EXECUCAO
color 0C
echo.
echo  ==================================================================================================
echo   ERRO DETECTADO!
echo  ==================================================================================================
echo.
echo   O programa fechou inesperadamente. Verifique a mensagem de erro acima.
echo.
pause
exit