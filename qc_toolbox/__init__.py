
__version__ = "1.0.0"
__author__ = "OSIPI — Open Science Initiative for Perfusion Imaging"


class QCToolboxError(Exception): pass


class BIDSLoadError(QCToolboxError): pass


class QCComputationError(QCToolboxError): pass


class ThresholdError(QCToolboxError): pass


class ReportError(QCToolboxError): pass
