## Règles sur les données (obligatoires)
- Les dossiers data/ et models/ contiennent des données MIMIC-IV soumises à une DUA PhysioNet.
- Ne jamais lire, afficher, échantillonner ou résumer leur contenu, par aucun moyen (outil de lecture, shell, script Python avec print).
- Les scripts qui traitent ces données ne doivent afficher que des statistiques agrégées (effectifs, pourcentages, quantiles), jamais d'extrait de texte ni de ligne.
- Pour tester le code, utiliser uniquement des données fictives dans tests/fixtures/.
- Ne jamais lancer toi-même un script sur les vraies données : donne-moi la commande et je l'exécute dans mon terminal.
