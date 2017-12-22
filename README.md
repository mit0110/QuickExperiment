# QuickExperiment

## To run test

You need the library nosetests. Test are oriented to check the model does not fail and
keeps certain properties. However, they do not check the correctness of the
implementation.

Examples:

```
QuickExperiment$ nosetests tests/
QuickExperiment$ nosetests tests/test_seq_lstm.py
QuickExperiment$ nosetests --nologcapture tests.test_seq_lstm:SeqLSTMModelTest.test_fit_loss
```