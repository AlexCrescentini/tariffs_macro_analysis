%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Global Trade Agent-Based Model (GTAB Model) - November 5th, 2025 %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Alex Crescentini
% Institution: IIASA - International Institute for Applied Systems Analysis
% Email: crescentini@iiasa.ac.at, crescentini.alex@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc; tic
main_path = fileparts(mfilename('fullpath'));
addpath(fullfile(main_path, 'data'));
addpath(fullfile(main_path, 'functions'));
addpath(fullfile(main_path, 'results'));
addpath(fullfile(main_path, 'shock'));
fprintf('Main path: %s\n', main_path);

%% Configuration (user inputs)

scenarios = [0]; %[0,1,2,3,4] 0 no tariffs, 1 USA vs CAN, 2 CAN retaliation, 3 LD, 4 World retaliation
MC = 250;
T = 12;
estimation_start = 2005;
cal_year_economy = 2024;
cal_year_IO = 2021;
scale = 1/1000;
fields_to_keep = {
    'nominal_gdp'
    'real_gdp'
    'nominal_household_consumption'
    'real_household_consumption'
    'log_avg_cpu_f'
    'log_avg_dpl_f'
    'log_avg_exp_f'
    'log_avg_cpu_fg'
    'log_avg_dpl_fg'
    'log_avg_exp_fg'
    'deflator_ppi_f'
    'deflator_ppi_fg'
    'dyn_bilateral_trade_g'
    'tariffs_paid'
};

%% Calibration

fprintf('Calibration...\n');
calibration = set_calibration(scenarios,MC,T,estimation_start,cal_year_economy,cal_year_IO,scale);
%save('./results/calibration.mat', 'calibration');

%% Simulation

fprintf('Simulation...'); 
for s = 1:length(scenarios)
    scenario = scenarios(s); 
    fprintf('\t scenario %d', scenario);
    tariff_shock = set_shock(calibration, scenario);
    out_simulated = cell(MC, 1);
    if MC == 1
        for mc = 1:MC
            rng(mc,'twister'); %"twister" guarantees results to be equal among two different matlab instances open, but both codes need to have "twister"
            [calibration_mc] = set_calibration_mc(calibration); 
            out_simulated{mc} = abm(tariff_shock, calibration_mc);
        end
    else
        parfor mc = 1:MC
            rng(mc,'twister'); %"twister" guarantees results to be equal among two different matlab instances open, but both codes need to have "twister"
            [calibration_mc] = set_calibration_mc(calibration); 
            out_simulated{mc} = abm(tariff_shock, calibration_mc);
        end
    end
    sim_result = get_results(MC, out_simulated);
    all_fields = fieldnames(sim_result);
    fields_to_remove = setdiff(all_fields, fields_to_keep);
    sim_result = rmfield(sim_result, fields_to_remove);
    save(sprintf('./results/sim_results_scenario%d.mat', scenario), 'sim_result', '-v7.3');
end
fprintf('\nSimulation completed! %.2f seconds!\n', toc);