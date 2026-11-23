from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ClientHello(_message.Message):
    __slots__ = ("cid",)
    CID_FIELD_NUMBER: _ClassVar[int]
    cid: int
    def __init__(self, cid: _Optional[int] = ...) -> None: ...

class ModelParameters(_message.Message):
    __slots__ = ("coef_values", "coef_shape", "intercept_values")
    COEF_VALUES_FIELD_NUMBER: _ClassVar[int]
    COEF_SHAPE_FIELD_NUMBER: _ClassVar[int]
    INTERCEPT_VALUES_FIELD_NUMBER: _ClassVar[int]
    coef_values: _containers.RepeatedScalarFieldContainer[float]
    coef_shape: _containers.RepeatedScalarFieldContainer[int]
    intercept_values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, coef_values: _Optional[_Iterable[float]] = ..., coef_shape: _Optional[_Iterable[int]] = ..., intercept_values: _Optional[_Iterable[float]] = ...) -> None: ...

class GlobalModel(_message.Message):
    __slots__ = ("round", "total_rounds", "done", "message", "model")
    ROUND_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ROUNDS_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    round: int
    total_rounds: int
    done: bool
    message: str
    model: ModelParameters
    def __init__(self, round: _Optional[int] = ..., total_rounds: _Optional[int] = ..., done: _Optional[bool] = ..., message: _Optional[str] = ..., model: _Optional[_Union[ModelParameters, _Mapping]] = ...) -> None: ...

class ClientUpdate(_message.Message):
    __slots__ = ("cid", "round", "num_examples", "train_loss", "train_acc", "model")
    CID_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    NUM_EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    TRAIN_LOSS_FIELD_NUMBER: _ClassVar[int]
    TRAIN_ACC_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    cid: int
    round: int
    num_examples: int
    train_loss: float
    train_acc: float
    model: ModelParameters
    def __init__(self, cid: _Optional[int] = ..., round: _Optional[int] = ..., num_examples: _Optional[int] = ..., train_loss: _Optional[float] = ..., train_acc: _Optional[float] = ..., model: _Optional[_Union[ModelParameters, _Mapping]] = ...) -> None: ...

class UpdateAck(_message.Message):
    __slots__ = ("accepted", "message", "server_round")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SERVER_ROUND_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    message: str
    server_round: int
    def __init__(self, accepted: _Optional[bool] = ..., message: _Optional[str] = ..., server_round: _Optional[int] = ...) -> None: ...
