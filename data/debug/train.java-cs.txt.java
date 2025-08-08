public ListSpeechSynthesisTasksResult listSpeechSynthesisTasks(ListSpeechSynthesisTasksRequest request) {request = beforeClientExecution(request);return executeListSpeechSynthesisTasks(request);}
public UpdateJourneyStateResult updateJourneyState(UpdateJourneyStateRequest request) {request = beforeClientExecution(request);return executeUpdateJourneyState(request);}
public void removePresentationFormat() {remove1stProperty(PropertyIDMap.PID_PRESFORMAT);}
public CellRangeAddressList(int firstRow, int lastRow, int firstCol, int lastCol) {this();addCellRangeAddress(firstRow, firstCol, lastRow, lastCol);}
public void delete(int key) {int i = binarySearch(mKeys, 0, mSize, key);if (i >= 0) {if (mValues[i] != DELETED) {mValues[i] = DELETED;mGarbage = true;}}}
public CreateBranchCommand setStartPoint(RevCommit startPoint) {checkCallable();this.startCommit = startPoint;this.startPoint = null;return this;}
public int centerX() {return x + w / 2;}
public ListPresetsResult listPresets() {return listPresets(new ListPresetsRequest());}
public DeleteFolderContentsResult deleteFolderContents(DeleteFolderContentsRequest request) {request = beforeClientExecution(request);return executeDeleteFolderContents(request);}
public GetConsoleOutputResult getConsoleOutput(GetConsoleOutputRequest request) {request = beforeClientExecution(request);return executeGetConsoleOutput(request);}
public PutMailboxPermissionsResult putMailboxPermissions(PutMailboxPermissionsRequest request) {request = beforeClientExecution(request);return executePutMailboxPermissions(request);}
public Cluster disableSnapshotCopy(DisableSnapshotCopyRequest request) {request = beforeClientExecution(request);return executeDisableSnapshotCopy(request);}
public static String stripExtension(String filename) {int idx = filename.indexOf('.');if (idx != -1) {filename = filename.substring(0, idx);}return filename;}
public ByteBuffer putInt(int value) {throw new ReadOnlyBufferException();}
public int lastIndexOf(final int o){int rval = _limit - 1;for (; rval >= 0; rval--){if (o == _array[ rval ]){break;}}return rval;}
public void setCountsByTime(int[] counts, long msecStep) {countsByTime = counts;countsByTimeStepMSec = msecStep;}
public FeatHdrRecord(RecordInputStream in) {futureHeader = new FtrHeader(in);isf_sharedFeatureType = in.readShort();reserved = in.readByte();cbHdrData = in.readInt();rgbHdrData = in.readRemainder();}
public CopyOnWriteArrayList() {elements = EmptyArray.OBJECT;}
public WriteRequest(DeleteRequest deleteRequest) {setDeleteRequest(deleteRequest);}
public void readFully(byte[] buf){_in.readFully(buf);}
