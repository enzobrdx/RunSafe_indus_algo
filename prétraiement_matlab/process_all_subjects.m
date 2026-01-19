
% --- Configuration ---
json_root_folder = 'C:/Users/Lenovo/OneDrive - Early Makers Group/Documents/Etudes/EMSE/RunSafe/Indus algo/ric_data/reformat_data';
output_csv_file = 'runsafe_processed_data.csv';

% Initialiser un cell array pour stocker les résultats (plus rapide que d'agrandir une table)
resultsCell = {};

% --- 1. Boucle sur les dossiers Sujet (sub_id) ---
subject_folders = dir(fullfile(json_root_folder, '*'));

fprintf('Début du traitement de %d sujets...\n', length(subject_folders));

for i = 1:length(subject_folders)
    subject_name = subject_folders(i).name;
    subject_path = fullfile(json_root_folder, subject_name);
    
    % --- 2. Boucle sur les fichiers JSON (sessions) de ce sujet ---
    json_files = dir(fullfile(subject_path, '*.json'));
    
    fprintf('  Sujet %s: %d sessions trouvées.\n', subject_name, length(json_files));
    
    for j = 1:length(json_files)
        session_name = json_files(j).name;
        session_path = fullfile(subject_path, session_name);
        
        fprintf('    Traitement de: %s\n', session_name);
        
        % --- 3. Traitement d'une seule session ---
        try
            % Charger et décoder le fichier JSON
            fid = fopen(session_path);
            raw = fread(fid,inf);
            str = char(raw');
            fclose(fid);
            data = jsondecode(str);

            fields = fieldnames(data.joints);
             
            for m = 1:size(fields,1)
                
                data.joints.(fields{m,1}) = transpose(data.joints.(fields{m,1}));
                
            end
            
            clear fields
            fields = fieldnames(data.neutral);
            
            for m = 1:size(fields,1)
                
                data.neutral.(fields{m,1}) = transpose(data.neutral.(fields{m,1}));
                
            end
            
            % Vérifier que les données de course existent
            if ~isfield(data, 'running') || ~isfield(data, 'hz_r') || isempty(data.running)
                % MODIFICATION: Ajout de '|| isempty(data.running)'
                fprintf('      Attention: Données "running" ou "hz_r" manquantes OU "running" est vide. Session ignorée.\n');
                % MODIFICATION: Message mis à jour
                continue;
            end

            % Étape 1: Calculer les angles et vitesses continus
            [angles, velocities, ~, ~, ~] = gait_kinematics(data.joints, data.neutral, data.running, data.hz_r, 0);
            
            % Étape 2: Calculer les métriques discrètes (médiane + std)
            % On utilise notre script modifié
            [session_metrics] = gait_steps_runsafe(data.neutral, data.running, angles, velocities, data.hz_r, 0);
            
            % Ajouter les infos d'identification
            session_metrics.sub_id = string(subject_name);
            session_metrics.session_file = string(session_name);
            
            % Ajouter la structure de résultats à notre cell array
            resultsCell{end+1} = session_metrics;
            
        catch ME
            % En cas d'échec sur un fichier (données corrompues, etc.)
            fprintf('      ERREUR lors du traitement de %s:\n', session_name);
            fprintf('        Message: %s\n', ME.message);
            
            % Afficher la pile d'appels (stack trace) pour trouver la ligne exacte
            if ~isempty(ME.stack)
                % L'erreur s'est produite dans le fichier ME.stack(1).file à la ligne ME.stack(1).line
                fprintf('        Erreur détectée dans le fichier: %s\n', ME.stack(1).file);
                fprintf('        À la ligne: %d\n', ME.stack(1).line);
                
                % Optionnel: Afficher toute la pile pour plus de contexte
                % disp('        Pile d''appels complète:');
                % disp(ME.stack); 
            end
        end
    end
end

% --- 4. Conversion finale et sauvegarde en CSV ---
if ~isempty(resultsCell)
    % Convertir le cell array de structures en une table unique
    resultsTable = struct2table([resultsCell{:}]);
    
    % Réorganiser les colonnes (mettre sub_id et session_file en premier)
    resultsTable = movevars(resultsTable, {'sub_id', 'session_file'}, 'Before', 1);
    
    % Écrire la table dans un fichier CSV
    writetable(resultsTable, output_csv_file);
    fprintf('\nTraitement terminé. Fichier CSV sauvegardé sous: %s\n', output_csv_file);
else
    fprintf('\nAucune donnée n''a été traitée. Aucun fichier CSV généré.\n');
end