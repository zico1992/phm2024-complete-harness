"""CLI for freezing, CSV replay and retrospective operational shadow evaluation."""
import argparse
from pathlib import Path
import json
import pandas as pd
from phm.data import load_x, write_json
from operational.core import freeze, Predictor, operational_input, labelled_join, evaluate_shadow, new_dir, sha, now


def execute(args):
    if args.command=='freeze':
        return freeze(args.model,args.training_csv,args.output,args.name,args.reference_limit,getattr(args,'environment',None))
    if args.command=='doctor':
        from operational.runtime import check_model_environment, check_baseline_environment
        return check_baseline_environment(args.baseline) if args.baseline else check_model_environment(args.model,args.environment)
    if args.command=='evaluate':
        from operational.delayed import delayed_evaluate
        return delayed_evaluate(args.baseline,args.run_dir,args.labels,args.output)
    if args.command=='verify-evaluation':
        from operational.delayed import verify_delayed
        return verify_delayed(args.baseline,args.run_dir,args.labels)
    if args.command=='serve':
        from operational.server import serve
        return serve(args.run_dir,args.baseline,args.port)
    predictor = Predictor(args.baseline)
    if args.command=='intake':
        d = operational_input(args.input,args.contract,predictor)
        print(json.dumps({'status':'valid','n':len(d),'engines':d.engine_id.nunique(),'flights':d[['engine_id','flight_id']].drop_duplicates().shape[0]}))
        return
    mode = 'shadow' if args.command=='shadow' else 'replay'
    if mode=='shadow':
        d = operational_input(args.input,args.contract,predictor)
    else:
        d = load_x(args.input)
    out = new_dir(args.output)
    predictions = predictor.predict(d,mode)
    # Commit predictions before opening any outcome labels.
    predictions.to_csv(out/'predictions.csv',index=False)
    ledger = {'created_at':now(),'predictions_saved_at':now(),'mode':mode,'baseline':predictor.manifest,
              'input_sha256':sha(args.input),'prediction_sha256':sha(out/'predictions.csv'),
              'workflow':('retrospective labelled shadow' if args.labels else 'operational shadow predictions; waiting for labels') if mode=='shadow' else 'CSV replay; row order has no temporal meaning',
              'operational_data_available':mode=='shadow','n':len(d),'labels_used_for_predictions':False}
    if mode=='shadow':
        ledger['contract_sha256'] = sha(args.contract)
    write_json(out/'run_manifest.json',ledger)
    if args.labels:
        joined = labelled_join(predictions,args.labels,mode)
        joined.to_csv(out/'evaluated_predictions.csv',index=False)
        report = evaluate_shadow(joined,predictor.manifest['threshold'])
        ledger['label_sha256'] = sha(args.labels)
        ledger['evaluated_predictions_sha256'] = sha(out/'evaluated_predictions.csv')
    else:
        report = {'observations':len(d),'labelled':0,'metrics':None,'status':'waiting for confirmed labels'}
    report['baseline_name'] = predictor.manifest['name']
    report['mode'] = mode
    write_json(out/'evaluation.json',report)
    ledger['evaluation_sha256'] = sha(out/'evaluation.json')
    write_json(out/'run_manifest.json',ledger)
    write_json(out/'completion.json',{'completed_at':now(),'mode':mode,'n':len(d),
        'prediction_sha256':sha(out/'predictions.csv'),'baseline_sha256':predictor.manifest['baseline_sha256']})
    predictor.assert_frozen()
    return {'output':str(out),'n':len(d),'labelled':report['labelled'],'mode':mode}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--model',required=True,help='Trusted saved model from the full baseline run')
    f.add_argument('--training-csv',required=True)
    f.add_argument('--output',required=True)
    f.add_argument('--name',required=True,help='Use an explicit sample/demo name for sample artifacts')
    f.add_argument('--reference-limit',type=int,default=50000)
    f.add_argument('--environment',help='Original training environment.json when not next to the model')
    d=sub.add_parser('doctor',help='JSON-only runtime preflight; does not load model')
    choice=d.add_mutually_exclusive_group(required=True)
    choice.add_argument('--model');choice.add_argument('--baseline')
    d.add_argument('--environment')
    for name in ['evaluate','verify-evaluation']:
        e=sub.add_parser(name)
        e.add_argument('--baseline',required=True);e.add_argument('--run-dir',required=True)
        e.add_argument('--labels',required=name=='evaluate')
        if name=='evaluate':e.add_argument('--output',required=True)
    for name in ['replay','shadow','intake']:
        s = sub.add_parser(name)
        s.add_argument('--baseline',required=True)
        s.add_argument('--input',required=True)
        if name in ['shadow','intake']:
            s.add_argument('--contract',required=True)
        if name!='intake':
            s.add_argument('--labels',help='Optional confirmed labels; otherwise attach later with evaluate')
            s.add_argument('--output',required=True)
    s = sub.add_parser('serve')
    s.add_argument('--run-dir',required=True)
    s.add_argument('--baseline',required=True)
    s.add_argument('--port',type=int,default=8765)
    return p


if __name__=='__main__':
    try:
        result = execute(parser().parse_args())
        if result is not None:
            print(json.dumps(result,indent=2))
    except (ValueError,FileNotFoundError,KeyError) as exc:
        raise SystemExit(f'Pilot stopped: {exc}')
